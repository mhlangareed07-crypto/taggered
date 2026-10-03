"""
app.py — PDF Image Alt-Text Autotagger
=======================================
Streamlit app that:
  1. Accepts a PDF upload.
  2. Uses PyMuPDF (fitz) to find embedded images.
  3. Inspects the PDF structure tree to detect images missing /Alt text.
  4. Sends each image (as PNG) to Google Gemini for a short description.
  5. Writes the description back into the PDF's accessibility structure
     (existing /StructElem gets /Alt set; untagged images get a new
     /Figure structure element attached to the StructTreeRoot).
  6. Shows a report and offers the fixed PDF for download.

Setup
-----
    pip install streamlit pymupdf google-generativeai

Create `.streamlit/secrets.toml`:
    OPENAI_API_KEY = "your-gemini-api-key"   # Gemini key under this name, per spec

Run:
    streamlit run app.py

Notes / limitations (read before production use)
------------------------------------------------
* Best-effort tagging: PyMuPDF has no high-level structure-tree API, so this
  edits raw PDF objects. Validate output with PAC 3 / veraPDF for strict
  PDF/UA conformance. Complex trees (nested figures, MCID mixes) are handled
  heuristically.
* Inline images (BI/ID/EI operators) are not scanned — only image XObjects.
* Never paste real API keys into chat or code. Rotate any key that has been
  exposed, and keep it only in st.secrets or environment variables.
"""

import re
import time

import streamlit as st

try:
    import fitz  # PyMuPDF
except ImportError:
    st.error("PyMuPDF is not installed. Run:  pip install pymupdf")
    st.stop()

try:
    import google.generativeai as genai
except ImportError:
    st.error("google-generativeai is not installed. Run:  pip install google-generativeai")
    st.stop()

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
MODEL_NAME = "gemini-1.5-flash"   # fast + cheap; switch to "gemini-2.0-flash" if desired
MIN_DIM_PX = 10                   # images smaller than this are treated as decorative
REQUEST_DELAY = 0.15              # gentle pause between Gemini calls (rate-limit safety)

ALT_PROMPT = (
    "Write concise alt text (maximum 125 characters) describing this image "
    "for a PDF accessibility tag. Return ONLY the alt text itself — no quotes, "
    "no explanation, no 'Image of' prefix."
)


# --------------------------------------------------------------------------- #
# Secrets / model
# --------------------------------------------------------------------------- #
def get_api_key() -> str | None:
    """Read the Gemini API key from st.secrets.

    The spec stores the Gemini key under the name OPENAI_API_KEY, so that name
    is tried first; common alternatives are accepted as fallbacks.
    """
    for name in ("OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        try:
            if name in st.secrets:
                return st.secrets[name]
        except Exception:
            # No secrets file configured at all.
            pass
    return None


def build_model():
    """Configure Gemini and return a GenerativeModel, or None with an error shown."""
    key = get_api_key()
    if not key:
        st.error(
            "No API key found. Add one to `.streamlit/secrets.toml`:\n\n"
            'OPENAI_API_KEY = "your-gemini-api-key"'
        )
        return None
    genai.configure(api_key=key)
    return genai.GenerativeModel(MODEL_NAME)


# --------------------------------------------------------------------------- #
# Low-level PDF structure helpers (PyMuPDF raw-object layer)
# --------------------------------------------------------------------------- #
def pdf_unicode_string(text: str) -> str:
    """Encode text as a PDF hex string (UTF-16BE with BOM) — safe for /Alt."""
    data = b"\xfe\xff" + text.encode("utf-16-be")
    return "<" + data.hex().upper() + ">"


def ensure_struct_tree_root(doc: "fitz.Document") -> int:
    """Guarantee the catalog has /StructTreeRoot (+ /MarkInfo) and return its xref."""
    cat = doc.pdf_catalog()
    ktype, kval = doc.xref_get_key(cat, "StructTreeRoot")
    m = re.search(r"(\d+)\s+0\s+R", kval or "")
    if ktype not in ("null", "unknown") and m:
        return int(m.group(1))

    # No structure tree yet: create a minimal one and mark the PDF as tagged.
    strt = doc.get_new_xref()
    doc.update_object(strt, "<< /Type /StructTreeRoot /K [] >>")
    doc.xref_set_key(cat, "StructTreeRoot", f"{strt} 0 R")
    mk_type, _ = doc.xref_get_key(cat, "MarkInfo")
    if mk_type in ("null", "unknown"):
        doc.xref_set_key(cat, "MarkInfo", "<< /Marked true >>")
    return strt


def scan_structure(doc: "fitz.Document"):
    """One pass over all objects.

    Returns:
        elems:       set of xrefs of /StructElem objects
        objr_target: {objr_xref: image_xref} for every /OBJR object
    """
    elems, objr_target = set(), {}
    for x in range(1, doc.xref_length()):
        try:
            obj_str = doc.xref_object(x, compressed=True)
        except Exception:
            continue
        if "/OBJR" in obj_str:
            m = re.search(r"/Obj\s+(\d+)\s+0\s+R", obj_str)
            if m:
                objr_target[x] = int(m.group(1))
        if "/StructElem" in obj_str:
            elems.add(x)
    return elems, objr_target


def k_children(doc: "fitz.Document", elem: int):
    """Return (child_refs, inline_image_xrefs) found in a StructElem's /K entry."""
    try:
        ktype, kval = doc.xref_get_key(elem, "K")
    except Exception:
        return [], []
    if ktype in ("null", "unknown") or not kval:
        return [], []
    refs = [int(m.group(1)) for m in re.finditer(r"(\d+)\s+0\s+R", kval)]
    inline = (
        [int(m.group(1)) for m in re.finditer(r"/Obj\s+(\d+)\s+0\s+R", kval)]
        if "/OBJR" in kval
        else []
    )
    return refs, inline


def map_images_to_elems(doc: "fitz.Document", elems: set, objr_target: dict) -> dict:
    """Build {image_xref: [StructElem xrefs that tag it]}.

    Walks each element's /K subtree (BFS, depth-capped) resolving OBJR refs,
    including OBJR dicts written inline inside the /K array.
    """
    img_map: dict[int, list[int]] = {}
    for e in elems:
        visited = set()
        frontier = k_children(doc, e)[0] + k_children(doc, e)[1]
        depth = 0
        while frontier and depth < 4:  # depth cap keeps pathological trees cheap
            next_frontier = []
            for node in frontier:
                if node in visited:
                    continue
                visited.add(node)
                if node in objr_target:
                    img_map.setdefault(objr_target[node], []).append(e)
                elif node in elems:
                    refs, inline = k_children(doc, node)
                    next_frontier.extend(refs + inline)
            frontier = next_frontier
            depth += 1
    return img_map


def elem_has_alt(doc: "fitz.Document", elem: int) -> bool:
    """Heuristic check: does this StructElem already carry non-empty /Alt text?"""
    try:
        atype, aval = doc.xref_get_key(elem, "Alt")
    except Exception:
        return False
    if atype in ("null", "unknown") or aval in ("null", "none"):
        return False
    clean = aval.strip().strip("<>()\"' ")
    return clean != "" and clean.upper() not in ("FEFF",)  # BOM-only = empty


def append_ref_to_k_array(doc: "fitz.Document", owner_xref: int, ref: str):
    """Append '<ref> 0 R' to an object's /K array, preserving existing entries."""
    ktype, kval = doc.xref_get_key(owner_xref, "K")
    inner = (kval or "").strip()
    if inner.startswith("[") and inner.endswith("]"):
        inner = inner[1:-1].strip()
    if ktype in ("null", "unknown") or inner == "":
        doc.xref_set_key(owner_xref, "K", f"[ {ref} ]")
    else:
        doc.xref_set_key(owner_xref, "K", f"[ {inner} {ref} ]")


def create_figure_elem(doc: "fitz.Document", strt: int, img_xref: int) -> int:
    """Create a new /Figure StructElem (with an OBJR to the image) under the root."""
    ex = doc.get_new_xref()
    doc.update_object(
        ex,
        f"<< /Type /StructElem /S /Figure /P {strt} 0 R "
        f"/K << /Type /OBJR /Obj {img_xref} 0 R >> >>",
    )
    append_ref_to_k_array(doc, strt, f"{ex} 0 R")
    return ex


# --------------------------------------------------------------------------- #
# Image extraction + Gemini
# --------------------------------------------------------------------------- #
def image_png_bytes(doc: "fitz.Document", img_xref: int):
    """Extract an image as normalized PNG bytes; returns (png, width, height)."""
    pix = fitz.Pixmap(doc, img_xref)
    if pix.colorspace is not None and pix.colorspace.n > 3:  # CMYK etc. -> RGB
        pix = fitz.Pixmap(fitz.csRGB, pix)
    return pix.tobytes("png"), pix.width, pix.height


def gemini_alt_text(model, png_bytes: bytes) -> str:
    """Ask Gemini for short alt text; returns the cleaned string."""
    response = model.generate_content(
        [ALT_PROMPT, {"mime_type": "image/png", "data": png_bytes}]
    )
    return (response.text or "").strip().strip('"')


# --------------------------------------------------------------------------- #
# Core processing pipeline
# --------------------------------------------------------------------------- #
def process_pdf(doc: "fitz.Document", model, max_images: int):
    """Find images, tag the ones missing /Alt, and return a report.

    Returns (records, stats) where records is a list of dicts and stats
    summarizes counts for the UI metrics.
    """
    # -- 1. Collect unique image XObjects and the pages they appear on --------
    image_pages: dict[int, set] = {}
    for pno, page in enumerate(doc, start=1):
        for img in page.get_images(full=True):
            xref = img[0]
            if xref:  # 0 = inline/invalid reference
                image_pages.setdefault(xref, set()).add(pno)

    # -- 2. Map structure elements to images, find who is missing /Alt --------
    strt = ensure_struct_tree_root(doc)
    elems, objr_target = scan_structure(doc)
    img_map = map_images_to_elems(doc, elems, objr_target)

    def needs_alt(img_xref: int) -> bool:
        return not any(elem_has_alt(doc, e) for e in img_map.get(img_xref, []))

    pending = [x for x in image_pages if needs_alt(x)]
    pending.sort(key=lambda x: min(image_pages[x]))  # process in page order

    # -- 3. Generate + write alt text -----------------------------------------
    records = []
    tagged = errors = tiny_skipped = 0
    capped = pending[max_images:]
    todo = pending[:max_images]

    progress = st.progress(0.0, text="Analyzing images…")
    status = st.empty()

    for i, img_xref in enumerate(todo):
        pages_str = ", ".join(str(p) for p in sorted(image_pages[img_xref]))
        status.write(f"Image {i + 1}/{len(todo)} (page {pages_str}, xref {img_xref})…")
        try:
            png, w, h = image_png_bytes(doc, img_xref)
            if w < MIN_DIM_PX or h < MIN_DIM_PX:
                records.append(
                    {"Pages": pages_str, "Image xref": img_xref,
                     "Status": "Skipped (too small / decorative)", "Alt text": "—"}
                )
                tiny_skipped += 1
            else:
                alt = gemini_alt_text(model, png)
                # Write /Alt into every existing elem missing it; else create one.
                targets = [e for e in img_map.get(img_xref, []) if not elem_has_alt(doc, e)]
                if not targets:
                    targets = [create_figure_elem(doc, strt, img_xref)]
                for e in targets:
                    doc.xref_set_key(e, "Alt", pdf_unicode_string(alt))
                records.append(
                    {"Pages": pages_str, "Image xref": img_xref,
                     "Status": "Tagged", "Alt text": alt}
                )
                tagged += 1
            time.sleep(REQUEST_DELAY)
        except Exception as exc:  # keep going; report failures individually
            records.append(
                {"Pages": pages_str, "Image xref": img_xref,
                 "Status": "Error", "Alt text": str(exc)[:120]}
            )
            errors += 1
        progress.progress((i + 1) / len(todo), text=f"Processed {i + 1}/{len(todo)}")

    for img_xref in capped:
        records.append(
            {"Pages": ", ".join(map(str, sorted(image_pages[img_xref]))),
             "Image xref": img_xref, "Status": "Not processed (limit)", "Alt text": "—"}
        )

    progress.empty()
    status.empty()

    stats = {
        "found": len(image_pages),
        "needed": len(pending),
        "tagged": tagged,
        "errors": errors,
        "skipped_tiny": tiny_skipped,
        "already_ok": len(image_pages) - len(pending),
    }
    return records, stats


# --------------------------------------------------------------------------- #
# Streamlit UI
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="PDF Alt-Text Autotagger", page_icon="🖼️", layout="centered")

st.title("🖼️ PDF Alt-Text Autotagger")
st.caption(
    "Finds images missing alt text, asks Gemini for descriptions, and writes them "
    "into the PDF's accessibility structure. Best-effort tagging — validate "
    "critical documents with PAC 3 or veraPDF."
)

uploaded = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded is not None:
    max_images = st.sidebar.slider(
        "Max images to process", min_value=1, max_value=200, value=50,
        help="Caps Gemini API calls per run to control cost."
    )

    if st.button("🚀 Generate alt text & fix PDF", type="primary"):
        model = build_model()
        if model:
            try:
                doc = fitz.open(stream=uploaded.read(), filetype="pdf")
                records, stats = process_pdf(doc, model, max_images)

                # -- Report -------------------------------------------------
                st.subheader("📋 Report")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Images found", stats["found"])
                c2.metric("Needed alt text", stats["needed"])
                c3.metric("Tagged ✅", stats["tagged"])
                c4.metric("Errors ⚠️", stats["errors"])
                st.caption(
                    f"{stats['already_ok']} already had alt text · "
                    f"{stats['skipped_tiny']} skipped as decorative/tiny"
                )

                if records:
                    st.dataframe(records, use_container_width=True, hide_index=True)
                else:
                    st.success("No missing alt text found — this PDF is already covered. 🎉")

                # -- Download -----------------------------------------------
                out_bytes = doc.tobytes(garbage=3, deflate=True)
                base = uploaded.name.rsplit(".", 1)[0]
                st.download_button(
                    "⬇️ Download tagged PDF",
                    data=out_bytes,
                    file_name=f"{base}_alt_tagged.pdf",
                    mime="application/pdf",
                    disabled=(stats["tagged"] == 0),
                )
                doc.close()
            except Exception as exc:
                st.exception(f"Processing failed: {exc}")
else:
    st.info("👆 Upload a PDF to begin.")"""
app.py — PDF Image Alt-Text Autotagger
=======================================
Streamlit app that:
  1. Accepts a PDF upload.
  2. Uses PyMuPDF (fitz) to find embedded images.
  3. Inspects the PDF structure tree to detect images missing /Alt text.
  4. Sends each image (as PNG) to Google Gemini for a short description.
  5. Writes the description back into the PDF's accessibility structure
     (existing /StructElem gets /Alt set; untagged images get a new
     /Figure structure element attached to the StructTreeRoot).
  6. Shows a report and offers the fixed PDF for download.

Setup
-----
    pip install streamlit pymupdf google-generativeai

Create `.streamlit/secrets.toml`:
    OPENAI_API_KEY = "your-gemini-api-key"   # Gemini key under this name, per spec

Run:
    streamlit run app.py

Notes / limitations (read before production use)
------------------------------------------------
* Best-effort tagging: PyMuPDF has no high-level structure-tree API, so this
  edits raw PDF objects. Validate output with PAC 3 / veraPDF for strict
  PDF/UA conformance. Complex trees (nested figures, MCID mixes) are handled
  heuristically.
* Inline images (BI/ID/EI operators) are not scanned — only image XObjects.
* Never paste real API keys into chat or code. Rotate any key that has been
  exposed, and keep it only in st.secrets or environment variables.
"""

import re
import time

import streamlit as st

try:
    import fitz  # PyMuPDF
except ImportError:
    st.error("PyMuPDF is not installed. Run:  pip install pymupdf")
    st.stop()

try:
    import google.generativeai as genai
except ImportError:
    st.error("google-generativeai is not installed. Run:  pip install google-generativeai")
    st.stop()

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
MODEL_NAME = "gemini-1.5-flash"   # fast + cheap; switch to "gemini-2.0-flash" if desired
MIN_DIM_PX = 10                   # images smaller than this are treated as decorative
REQUEST_DELAY = 0.15              # gentle pause between Gemini calls (rate-limit safety)

ALT_PROMPT = (
    "Write concise alt text (maximum 125 characters) describing this image "
    "for a PDF accessibility tag. Return ONLY the alt text itself — no quotes, "
    "no explanation, no 'Image of' prefix."
)


# --------------------------------------------------------------------------- #
# Secrets / model
# --------------------------------------------------------------------------- #
def get_api_key() -> str | None:
    """Read the Gemini API key from st.secrets.

    The spec stores the Gemini key under the name OPENAI_API_KEY, so that name
    is tried first; common alternatives are accepted as fallbacks.
    """
    for name in ("OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        try:
            if name in st.secrets:
                return st.secrets[name]
        except Exception:
            # No secrets file configured at all.
            pass
    return None


def build_model():
    """Configure Gemini and return a GenerativeModel, or None with an error shown."""
    key = get_api_key()
    if not key:
        st.error(
            "No API key found. Add one to `.streamlit/secrets.toml`:\n\n"
            'OPENAI_API_KEY = "your-gemini-api-key"'
        )
        return None
    genai.configure(api_key=key)
    return genai.GenerativeModel(MODEL_NAME)


# --------------------------------------------------------------------------- #
# Low-level PDF structure helpers (PyMuPDF raw-object layer)
# --------------------------------------------------------------------------- #
def pdf_unicode_string(text: str) -> str:
    """Encode text as a PDF hex string (UTF-16BE with BOM) — safe for /Alt."""
    data = b"\xfe\xff" + text.encode("utf-16-be")
    return "<" + data.hex().upper() + ">"


def ensure_struct_tree_root(doc: "fitz.Document") -> int:
    """Guarantee the catalog has /StructTreeRoot (+ /MarkInfo) and return its xref."""
    cat = doc.pdf_catalog()
    ktype, kval = doc.xref_get_key(cat, "StructTreeRoot")
    m = re.search(r"(\d+)\s+0\s+R", kval or "")
    if ktype not in ("null", "unknown") and m:
        return int(m.group(1))

    # No structure tree yet: create a minimal one and mark the PDF as tagged.
    strt = doc.get_new_xref()
    doc.update_object(strt, "<< /Type /StructTreeRoot /K [] >>")
    doc.xref_set_key(cat, "StructTreeRoot", f"{strt} 0 R")
    mk_type, _ = doc.xref_get_key(cat, "MarkInfo")
    if mk_type in ("null", "unknown"):
        doc.xref_set_key(cat, "MarkInfo", "<< /Marked true >>")
    return strt


def scan_structure(doc: "fitz.Document"):
    """One pass over all objects.

    Returns:
        elems:       set of xrefs of /StructElem objects
        objr_target: {objr_xref: image_xref} for every /OBJR object
    """
    elems, objr_target = set(), {}
    for x in range(1, doc.xref_length()):
        try:
            obj_str = doc.xref_object(x, compressed=True)
        except Exception:
            continue
        if "/OBJR" in obj_str:
            m = re.search(r"/Obj\s+(\d+)\s+0\s+R", obj_str)
            if m:
                objr_target[x] = int(m.group(1))
        if "/StructElem" in obj_str:
            elems.add(x)
    return elems, objr_target


def k_children(doc: "fitz.Document", elem: int):
    """Return (child_refs, inline_image_xrefs) found in a StructElem's /K entry."""
    try:
        ktype, kval = doc.xref_get_key(elem, "K")
    except Exception:
        return [], []
    if ktype in ("null", "unknown") or not kval:
        return [], []
    refs = [int(m.group(1)) for m in re.finditer(r"(\d+)\s+0\s+R", kval)]
    inline = (
        [int(m.group(1)) for m in re.finditer(r"/Obj\s+(\d+)\s+0\s+R", kval)]
        if "/OBJR" in kval
        else []
    )
    return refs, inline


def map_images_to_elems(doc: "fitz.Document", elems: set, objr_target: dict) -> dict:
    """Build {image_xref: [StructElem xrefs that tag it]}.

    Walks each element's /K subtree (BFS, depth-capped) resolving OBJR refs,
    including OBJR dicts written inline inside the /K array.
    """
    img_map: dict[int, list[int]] = {}
    for e in elems:
        visited = set()
        frontier = k_children(doc, e)[0] + k_children(doc, e)[1]
        depth = 0
        while frontier and depth < 4:  # depth cap keeps pathological trees cheap
            next_frontier = []
            for node in frontier:
                if node in visited:
                    continue
                visited.add(node)
                if node in objr_target:
                    img_map.setdefault(objr_target[node], []).append(e)
                elif node in elems:
                    refs, inline = k_children(doc, node)
                    next_frontier.extend(refs + inline)
            frontier = next_frontier
            depth += 1
    return img_map


def elem_has_alt(doc: "fitz.Document", elem: int) -> bool:
    """Heuristic check: does this StructElem already carry non-empty /Alt text?"""
    try:
        atype, aval = doc.xref_get_key(elem, "Alt")
    except Exception:
        return False
    if atype in ("null", "unknown") or aval in ("null", "none"):
        return False
    clean = aval.strip().strip("<>()\"' ")
    return clean != "" and clean.upper() not in ("FEFF",)  # BOM-only = empty


def append_ref_to_k_array(doc: "fitz.Document", owner_xref: int, ref: str):
    """Append '<ref> 0 R' to an object's /K array, preserving existing entries."""
    ktype, kval = doc.xref_get_key(owner_xref, "K")
    inner = (kval or "").strip()
    if inner.startswith("[") and inner.endswith("]"):
        inner = inner[1:-1].strip()
    if ktype in ("null", "unknown") or inner == "":
        doc.xref_set_key(owner_xref, "K", f"[ {ref} ]")
    else:
        doc.xref_set_key(owner_xref, "K", f"[ {inner} {ref} ]")


def create_figure_elem(doc: "fitz.Document", strt: int, img_xref: int) -> int:
    """Create a new /Figure StructElem (with an OBJR to the image) under the root."""
    ex = doc.get_new_xref()
    doc.update_object(
        ex,
        f"<< /Type /StructElem /S /Figure /P {strt} 0 R "
        f"/K << /Type /OBJR /Obj {img_xref} 0 R >> >>",
    )
    append_ref_to_k_array(doc, strt, f"{ex} 0 R")
    return ex


# --------------------------------------------------------------------------- #
# Image extraction + Gemini
# --------------------------------------------------------------------------- #
def image_png_bytes(doc: "fitz.Document", img_xref: int):
    """Extract an image as normalized PNG bytes; returns (png, width, height)."""
    pix = fitz.Pixmap(doc, img_xref)
    if pix.colorspace is not None and pix.colorspace.n > 3:  # CMYK etc. -> RGB
        pix = fitz.Pixmap(fitz.csRGB, pix)
    return pix.tobytes("png"), pix.width, pix.height


def gemini_alt_text(model, png_bytes: bytes) -> str:
    """Ask Gemini for short alt text; returns the cleaned string."""
    response = model.generate_content(
        [ALT_PROMPT, {"mime_type": "image/png", "data": png_bytes}]
    )
    return (response.text or "").strip().strip('"')


# --------------------------------------------------------------------------- #
# Core processing pipeline
# --------------------------------------------------------------------------- #
def process_pdf(doc: "fitz.Document", model, max_images: int):
    """Find images, tag the ones missing /Alt, and return a report.

    Returns (records, stats) where records is a list of dicts and stats
    summarizes counts for the UI metrics.
    """
    # -- 1. Collect unique image XObjects and the pages they appear on --------
    image_pages: dict[int, set] = {}
    for pno, page in enumerate(doc, start=1):
        for img in page.get_images(full=True):
            xref = img[0]
            if xref:  # 0 = inline/invalid reference
                image_pages.setdefault(xref, set()).add(pno)

    # -- 2. Map structure elements to images, find who is missing /Alt --------
    strt = ensure_struct_tree_root(doc)
    elems, objr_target = scan_structure(doc)
    img_map = map_images_to_elems(doc, elems, objr_target)

    def needs_alt(img_xref: int) -> bool:
        return not any(elem_has_alt(doc, e) for e in img_map.get(img_xref, []))

    pending = [x for x in image_pages if needs_alt(x)]
    pending.sort(key=lambda x: min(image_pages[x]))  # process in page order

    # -- 3. Generate + write alt text -----------------------------------------
    records = []
    tagged = errors = tiny_skipped = 0
    capped = pending[max_images:]
    todo = pending[:max_images]

    progress = st.progress(0.0, text="Analyzing images…")
    status = st.empty()

    for i, img_xref in enumerate(todo):
        pages_str = ", ".join(str(p) for p in sorted(image_pages[img_xref]))
        status.write(f"Image {i + 1}/{len(todo)} (page {pages_str}, xref {img_xref})…")
        try:
            png, w, h = image_png_bytes(doc, img_xref)
            if w < MIN_DIM_PX or h < MIN_DIM_PX:
                records.append(
                    {"Pages": pages_str, "Image xref": img_xref,
                     "Status": "Skipped (too small / decorative)", "Alt text": "—"}
                )
                tiny_skipped += 1
            else:
                alt = gemini_alt_text(model, png)
                # Write /Alt into every existing elem missing it; else create one.
                targets = [e for e in img_map.get(img_xref, []) if not elem_has_alt(doc, e)]
                if not targets:
                    targets = [create_figure_elem(doc, strt, img_xref)]
                for e in targets:
                    doc.xref_set_key(e, "Alt", pdf_unicode_string(alt))
                records.append(
                    {"Pages": pages_str, "Image xref": img_xref,
                     "Status": "Tagged", "Alt text": alt}
                )
                tagged += 1
            time.sleep(REQUEST_DELAY)
        except Exception as exc:  # keep going; report failures individually
            records.append(
                {"Pages": pages_str, "Image xref": img_xref,
                 "Status": "Error", "Alt text": str(exc)[:120]}
            )
            errors += 1
        progress.progress((i + 1) / len(todo), text=f"Processed {i + 1}/{len(todo)}")

    for img_xref in capped:
        records.append(
            {"Pages": ", ".join(map(str, sorted(image_pages[img_xref]))),
             "Image xref": img_xref, "Status": "Not processed (limit)", "Alt text": "—"}
        )

    progress.empty()
    status.empty()

    stats = {
        "found": len(image_pages),
        "needed": len(pending),
        "tagged": tagged,
        "errors": errors,
        "skipped_tiny": tiny_skipped,
        "already_ok": len(image_pages) - len(pending),
    }
    return records, stats


# --------------------------------------------------------------------------- #
# Streamlit UI
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="PDF Alt-Text Autotagger", page_icon="🖼️", layout="centered")

st.title("🖼️ PDF Alt-Text Autotagger")
st.caption(
    "Finds images missing alt text, asks Gemini for descriptions, and writes them "
    "into the PDF's accessibility structure. Best-effort tagging — validate "
    "critical documents with PAC 3 or veraPDF."
)

uploaded = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded is not None:
    max_images = st.sidebar.slider(
        "Max images to process", min_value=1, max_value=200, value=50,
        help="Caps Gemini API calls per run to control cost."
    )

    if st.button("🚀 Generate alt text & fix PDF", type="primary"):
        model = build_model()
        if model:
            try:
                doc = fitz.open(stream=uploaded.read(), filetype="pdf")
                records, stats = process_pdf(doc, model, max_images)

                # -- Report -------------------------------------------------
                st.subheader("📋 Report")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Images found", stats["found"])
                c2.metric("Needed alt text", stats["needed"])
                c3.metric("Tagged ✅", stats["tagged"])
                c4.metric("Errors ⚠️", stats["errors"])
                st.caption(
                    f"{stats['already_ok']} already had alt text · "
                    f"{stats['skipped_tiny']} skipped as decorative/tiny"
                )

                if records:
                    st.dataframe(records, use_container_width=True, hide_index=True)
                else:
                    st.success("No missing alt text found — this PDF is already covered. 🎉")

                # -- Download -----------------------------------------------
                out_bytes = doc.tobytes(garbage=3, deflate=True)
                base = uploaded.name.rsplit(".", 1)[0]
                st.download_button(
                    "⬇️ Download tagged PDF",
                    data=out_bytes,
                    file_name=f"{base}_alt_tagged.pdf",
                    mime="application/pdf",
                    disabled=(stats["tagged"] == 0),
                )
                doc.close()
            except Exception as exc:
                st.exception(f"Processing failed: {exc}")
else:
    st.info("👆 Upload a PDF to begin.")
