
import streamlit as st
import pandas as pd
import numpy as np
import re
import io
import json
import datetime
import uuid
from difflib import SequenceMatcher
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch

st.set_page_config(page_title="Quote-to-PO Demo (v2.1)", layout="wide")

CURRENCIES = ["AED", "SAR", "QAR", "OMR", "BHD", "KWD", "USD", "EUR"]

def now_iso():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def normalize_desc(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\u0600-\u06FF\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def sim(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_desc(a), normalize_desc(b)).ratio()

def safe_float(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if not s or s.lower() in ["nan", "none", "null"]:
        return None
    s = s.replace(",", "")
    s = re.sub(r"[^\d\.\-]", "", s)
    try:
        return float(s)
    except Exception:
        return None

def is_blank_desc(v):
    if v is None:
        return True
    try:
        if pd.isna(v):
            return True
    except Exception:
        pass
    s = str(v).strip()
    return (s == "" or s.lower() in ["nan", "none", "null"])

def detect_columns(cols):
    cols_l = {str(c).strip().lower(): c for c in cols}
    synonyms = {
        "description": ["description", "item", "items", "product", "material", "service", "particular", "particulars", "desc", "details", "work"],
        "qty": ["qty", "quantity", "q'ty", "qtty", "quan", "q"],
        "unit": ["unit", "uom", "measure", "um"],
        "unit_price": ["unit price", "rate", "price", "unit_rate", "unitrate", "unitprice"],
        "total": ["amount", "total", "line total", "line_total", "value", "subtotal", "ext price", "extended"]
    }
    mapping = {}
    for k, opts in synonyms.items():
        for opt in opts:
            if opt in cols_l:
                mapping[k] = cols_l[opt]
                break
    return mapping

def df_to_items(df: pd.DataFrame, colmap: dict):
    items = []
    for _, r in df.iterrows():
        desc_raw = r.get(colmap.get("description", ""), "")
        desc = "" if is_blank_desc(desc_raw) else str(desc_raw).strip()

        qty = safe_float(r.get(colmap.get("qty", ""), None))
        unit = r.get(colmap.get("unit", ""), "")
        unit = "" if is_blank_desc(unit) else str(unit).strip()

        unit_price = safe_float(r.get(colmap.get("unit_price", ""), None))
        total = safe_float(r.get(colmap.get("total", ""), None))

        if desc and desc.upper() in ["CLIENT DETAILS", "QUOTATION", "QUOTATION DETAILS", "SCOPE SUMMARY"]:
            continue

        if desc == "" and qty is None and unit_price is None and total is None:
            continue

        if total is None and qty is not None and unit_price is not None:
            total = qty * unit_price

        if desc == "" and not (qty is not None and unit_price is not None):
            continue

        items.append({
            "description": desc,
            "qty": qty,
            "unit": unit,
            "unit_price": unit_price,
            "total": total
        })
    return items

def parse_pasted_table(text: str):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    items = []
    for ln in lines:
        if "|" in ln:
            parts = [p.strip() for p in ln.split("|")]
        elif "\t" in ln:
            parts = [p.strip() for p in ln.split("\t")]
        else:
            parts = [p.strip() for p in ln.split(",")]
        parts += [""] * (5 - len(parts))
        desc, qty, unit, unit_price, total = parts[:5]
        items.append({
            "description": desc,
            "qty": safe_float(qty),
            "unit": unit,
            "unit_price": safe_float(unit_price),
            "total": safe_float(total)
        })
    for it in items:
        if it.get("total") is None and it.get("qty") is not None and it.get("unit_price") is not None:
            it["total"] = it["qty"] * it["unit_price"]
    return items

def extract_text_from_pdf(file_bytes: bytes):
    txt = ""
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages[:10]:
                t = page.extract_text() or ""
                txt += t + "\n"
    except Exception:
        txt = ""
    if not txt.strip():
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            for page in reader.pages[:10]:
                txt += (page.extract_text() or "") + "\n"
        except Exception:
            pass
    return txt.strip()

def ocr_image_bytes(image_bytes: bytes, lang: str = "eng"):
    try:
        from PIL import Image
        import pytesseract
    except Exception as e:
        return "", f"OCR dependencies missing: {e}"
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        text = pytesseract.image_to_string(img, lang=lang)
        return text.strip(), None
    except Exception as e:
        return "", str(e)

def ocr_pdf_bytes(pdf_bytes: bytes, lang: str = "eng", max_pages: int = 5):
    try:
        from pdf2image import convert_from_bytes
        import pytesseract
    except Exception as e:
        return "", f"OCR dependencies missing: {e}"
    try:
        images = convert_from_bytes(pdf_bytes, first_page=1, last_page=max_pages, fmt="png")
        text_parts = []
        for img in images:
            text_parts.append(pytesseract.image_to_string(img, lang=lang))
        return "\n".join(text_parts).strip(), None
    except Exception as e:
        return "", str(e)

def heuristic_items_from_text(txt: str):
    items = []
    blocklist = ["client details", "quotation", "scope summary", "email", "mobile", "site / location"]
    for ln in txt.splitlines():
        line = re.sub(r"\s+", " ", ln).strip()
        if len(line) < 8:
            continue
        low = line.lower()
        if any(b in low for b in blocklist):
            continue
        nums = re.findall(r"[-+]?\d+(?:\.\d+)?", line.replace(",", ""))
        if len(nums) < 2:
            continue
        unit_price = safe_float(nums[-2]) if len(nums) >= 2 else None
        total = safe_float(nums[-1]) if len(nums) >= 1 else None
        qty = safe_float(nums[0])
        desc = re.sub(r"[-+]?\d+(?:\.\d+)?", " ", line)
        desc = re.sub(r"\s+", " ", desc).strip()
        if len(desc) < 3:
            continue
        if qty is None and unit_price is None and total is None:
            continue
        items.append({"description": desc, "qty": qty, "unit": "", "unit_price": unit_price, "total": total})
    return items[:80]

def build_comparison(quotes):
    canon = []
    for q in quotes:
        for it in q["items"]:
            d = str(it.get("description","")).strip()
            if d:
                canon.append(d)
    clusters = []
    for d in canon:
        placed = False
        for c in clusters:
            if sim(d, c["key"]) >= 0.88:
                c["alts"].append(d)
                placed = True
                break
        if not placed:
            clusters.append({"key": d, "alts": [d]})
    rows = []
    for c in clusters:
        row = {"item": c["key"]}
        for q in quotes:
            best = None
            best_score = 0.0
            for it in q["items"]:
                score = sim(c["key"], it.get("description",""))
                if score > best_score:
                    best_score = score
                    best = it
            if best is not None and best_score >= 0.80:
                row[f'{q["vendor"]} | unit_price'] = best.get("unit_price")
                row[f'{q["vendor"]} | total'] = best.get("total")
                row[f'{q["vendor"]} | qty'] = best.get("qty")
            else:
                row[f'{q["vendor"]} | unit_price'] = None
                row[f'{q["vendor"]} | total'] = None
                row[f'{q["vendor"]} | qty'] = None
        rows.append(row)
    return pd.DataFrame(rows)

def add_flags(df: pd.DataFrame, vendors):
    flags = []
    for _, r in df.iterrows():
        prices = []
        for v in vendors:
            p = r.get(f"{v} | unit_price", None)
            if p is not None and not (isinstance(p, float) and np.isnan(p)):
                prices.append(float(p))
        if not prices:
            continue
        mn = min(prices)
        for v in vendors:
            p = r.get(f"{v} | unit_price", None)
            if p is None or (isinstance(p, float) and np.isnan(p)):
                flags.append((r["item"], v, "missing"))
            else:
                if mn > 0 and float(p) > mn * 1.15:
                    flags.append((r["item"], v, f"outlier >15% vs min ({mn:.2f})"))
    return flags

def generate_po_pdf(company_name, vendor, po_number, items, currency="AED"):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    w, h = letter
    x = 0.7*inch
    y = h - 0.7*inch
    c.setFont("Helvetica-Bold", 16)
    c.drawString(x, y, "PURCHASE ORDER")
    y -= 0.25*inch
    c.setFont("Helvetica", 10)
    c.drawString(x, y, f"Company: {company_name}")
    y -= 0.18*inch
    c.drawString(x, y, f"Vendor: {vendor}")
    y -= 0.18*inch
    c.drawString(x, y, f"PO No: {po_number}    Date: {datetime.date.today().isoformat()}")
    y -= 0.28*inch

    c.setFont("Helvetica-Bold", 10)
    c.drawString(x, y, "Description")
    c.drawString(x+4.0*inch, y, "Qty")
    c.drawString(x+4.8*inch, y, "Unit Price")
    c.drawString(x+6.2*inch, y, "Total")
    y -= 0.12*inch
    c.line(x, y, w-0.7*inch, y)
    y -= 0.12*inch

    c.setFont("Helvetica", 9)
    grand = 0.0
    for it in items:
        desc = (it.get("description") or "")[:60]
        qty = it.get("qty") or 0
        up = it.get("unit_price") or 0
        tot = it.get("total")
        if tot is None:
            tot = qty * up
        grand += float(tot or 0)
        c.drawString(x, y, desc)
        c.drawRightString(x+4.6*inch, y, f"{qty:g}")
        c.drawRightString(x+6.0*inch, y, f"{up:.2f}")
        c.drawRightString(x+7.2*inch, y, f"{tot:.2f}")
        y -= 0.16*inch
        if y < 1.2*inch:
            c.showPage()
            y = h - 0.7*inch
            c.setFont("Helvetica", 9)

    y -= 0.1*inch
    c.setFont("Helvetica-Bold", 10)
    c.drawRightString(x+7.2*inch, y, f"Grand Total ({currency}): {grand:.2f}")
    y -= 0.3*inch
    c.setFont("Helvetica", 9)
    c.drawString(x, y, "Notes: Demo PO generated by Quote-to-PO OS (v2.1).")
    c.showPage()
    c.save()
    buf.seek(0)
    return buf

def find_header_row(raw: pd.DataFrame):
    keywords = ["description", "item", "particular", "qty", "quantity", "unit", "uom", "rate", "unit price", "price", "amount", "total"]
    best = (0, 0)
    for i in range(min(40, len(raw))):
        row = raw.iloc[i].astype(str).str.lower().fillna("")
        score = 0
        for cell in row.values:
            for k in keywords:
                if k in cell:
                    score += 1
        if score > best[1]:
            best = (i, score)
    return best[0], best[1]

# ---------------------------
# State
# ---------------------------
if "quotes" not in st.session_state:
    st.session_state.quotes = []
if "company" not in st.session_state:
    st.session_state.company = "Demo Company LLC"

st.title("Quote-to-PO OS — Interactive Demo (v2.1)")
st.caption("Fixes: duplicate widget IDs + better Excel parsing (header-row selection) + filters noisy rows.")

with st.sidebar:
    st.subheader("Demo Settings")
    st.session_state.company = st.text_input("Company name", st.session_state.company, key="company_name")
    use_ocr = st.toggle("Enable OCR for scanned PDFs/images", value=True, key="use_ocr")
    ocr_lang = st.selectbox("OCR language", ["eng", "ara+eng"], index=0, key="ocr_lang")
    st.markdown("---")
    st.info("Tip: For Excel exports with header blocks above items table, adjust the header-row index after upload.")

tab1, tab2, tab3, tab4 = st.tabs(["1) Add Quotes", "2) Review & Edit", "3) Compare", "4) Generate PO"])

# ---------------------------
# Tab 1: Add Quotes
# ---------------------------
with tab1:
    st.subheader("Add a supplier quote")
    colA, colB = st.columns([2, 1])
    with colA:
        vendor = st.text_input("Vendor name", value=f"Vendor {len(st.session_state.quotes)+1}", key="vendor_add")
        currency = st.selectbox("Currency", CURRENCIES, index=0, key="currency_add")
        uploaded = st.file_uploader("Upload quote file (PDF, PNG/JPG, XLSX, CSV)", type=["pdf", "png", "jpg", "jpeg", "xlsx", "csv"], key="uploader_add")
    with colB:
        pasted = st.text_area("Paste line items (desc|qty|unit|unit_price|total)", height=170, key="pasted_add",
                              placeholder="Example:\nGypsum board 12mm | 50 | pcs | 22.5 |\nMetal stud 75mm | 100 | pcs | 8.0 |")

    items = []
    raw_text = ""
    raw_df = None

    if uploaded is not None:
        name = uploaded.name.lower()
        data = uploaded.getvalue()

        if name.endswith(".csv"):
            raw_df = pd.read_csv(io.BytesIO(data), header=None)
        elif name.endswith(".xlsx"):
            try:
                sheets = pd.read_excel(io.BytesIO(data), sheet_name=None, header=None)
                sheet_names = list(sheets.keys())
                sel_sheet = st.selectbox("Select sheet", sheet_names, key="sheet_select")
                raw_df = sheets[sel_sheet]
            except Exception:
                raw_df = pd.read_excel(io.BytesIO(data), header=None)
        elif name.endswith(".pdf"):
            raw_text = extract_text_from_pdf(data)
            if not raw_text and use_ocr:
                raw_text, err = ocr_pdf_bytes(data, lang=ocr_lang)
                if err:
                    st.warning(f"OCR PDF error: {err}")
            items = heuristic_items_from_text(raw_text) if raw_text else []
        elif name.endswith((".png", ".jpg", ".jpeg")):
            st.image(uploaded, caption="Uploaded image", use_container_width=True)
            if use_ocr:
                raw_text, err = ocr_image_bytes(data, lang=ocr_lang)
                if err:
                    st.warning(f"OCR image error: {err}")
                items = heuristic_items_from_text(raw_text) if raw_text else []

        if raw_df is not None:
            st.write("Raw spreadsheet preview (top 30 rows):")
            st.dataframe(raw_df.head(30), use_container_width=True)

            guess_row, guess_score = find_header_row(raw_df)
            header_row = st.number_input("Header row index (0-based)", min_value=0, max_value=max(0, len(raw_df)-1),
                                         value=int(guess_row), step=1, key="header_row")
            st.caption(f"Auto-guess score: {guess_score}. Adjust until the structured preview shows line-item columns.")

            hdr = raw_df.iloc[header_row].astype(str).tolist()
            df = raw_df.iloc[header_row+1:].copy()
            df.columns = [h.strip() if h.strip() != "" else f"col_{i}" for i, h in enumerate(hdr)]
            df = df.reset_index(drop=True)

            st.write("Structured table preview (after header row):")
            st.dataframe(df.head(30), use_container_width=True)

            colmap_guess = detect_columns(df.columns)
            cols = list(df.columns)

            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                desc_col = st.selectbox("Description", cols, index=cols.index(colmap_guess.get("description", cols[0])), key="map_desc")
            with c2:
                qty_col = st.selectbox("Qty", cols, index=cols.index(colmap_guess.get("qty", cols[min(1,len(cols)-1)])), key="map_qty")
            with c3:
                unit_col = st.selectbox("Unit", cols, index=cols.index(colmap_guess.get("unit", cols[min(2,len(cols)-1)])), key="map_unit")
            with c4:
                up_col = st.selectbox("Unit Price", cols, index=cols.index(colmap_guess.get("unit_price", cols[min(3,len(cols)-1)])), key="map_up")
            with c5:
                tot_col = st.selectbox("Total", cols, index=cols.index(colmap_guess.get("total", cols[min(4,len(cols)-1)])), key="map_total")

            colmap = {"description": desc_col, "qty": qty_col, "unit": unit_col, "unit_price": up_col, "total": tot_col}
            items = df_to_items(df, colmap)

        if raw_text:
            with st.expander("Extracted text (from PDF/OCR)"):
                st.text_area("Text", raw_text, height=220, key="text_view")

    if pasted.strip():
        items = parse_pasted_table(pasted)

    st.markdown("---")
    st.subheader("Preview extracted/entered line items")
    if items:
        df_items = pd.DataFrame(items)
        df_items["total"] = df_items.apply(lambda r: (r.get("qty") or 0)*(r.get("unit_price") or 0)
                                           if pd.isna(r.get("total")) or r.get("total") is None else r.get("total"), axis=1)
        st.dataframe(df_items, use_container_width=True)

        if st.button("Save this quote", type="primary", key="save_quote_btn"):
            q = {
                "id": str(uuid.uuid4())[:8],
                "created_at": now_iso(),
                "vendor": vendor.strip() or "Unknown Vendor",
                "currency": currency,
                "source_filename": uploaded.name if uploaded is not None else "pasted",
                "items": df_items.fillna("").to_dict(orient="records"),
                "status": "Draft"
            }
            st.session_state.quotes.append(q)
            st.success(f"Saved quote for {q['vendor']} with {len(q['items'])} items.")
    else:
        st.info("Upload a file or paste items to create a quote. For Excel exports, set the correct header row.")

# ---------------------------
# Tab 2: Review & Edit
# ---------------------------
with tab2:
    st.subheader("Review & edit quotes (human-in-the-loop)")
    if not st.session_state.quotes:
        st.info("No quotes yet. Add quotes in Tab 1.")
    else:
        qnames = [f'{q["vendor"]} • {q["source_filename"]} • {q["id"]}' for q in st.session_state.quotes]
        sel = st.selectbox("Select a quote", qnames, key="quote_select_edit")
        idx = qnames.index(sel)
        q = st.session_state.quotes[idx]

        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            q["vendor"] = st.text_input("Vendor", q["vendor"], key=f"vendor_edit_{q['id']}")
        with col2:
            q["currency"] = st.selectbox("Currency", CURRENCIES, index=CURRENCIES.index(q.get("currency","AED")), key=f"currency_edit_{q['id']}")
        with col3:
            q["status"] = st.selectbox("Status", ["Draft","Submitted","Approved","Rejected"],
                                       index=["Draft","Submitted","Approved","Rejected"].index(q.get("status","Draft")),
                                       key=f"status_{q['id']}")

        df = pd.DataFrame(q["items"])
        if df.empty:
            st.warning("No line items found.")
        else:
            for col in ["description","qty","unit","unit_price","total"]:
                if col not in df.columns:
                    df[col] = ""

            edited = st.data_editor(df[["description","qty","unit","unit_price","total"]],
                                    use_container_width=True, num_rows="dynamic", key=f"edit_table_{q['id']}")

            def coerce(v):
                try:
                    return float(str(v).replace(",",""))
                except:
                    return None

            edited2 = edited.copy()
            for irow in range(len(edited2)):
                qty = coerce(edited2.loc[irow, "qty"])
                up = coerce(edited2.loc[irow, "unit_price"])
                tot = coerce(edited2.loc[irow, "total"])
                if tot is None and qty is not None and up is not None:
                    edited2.loc[irow, "total"] = qty * up

            if st.button("Save edits", type="primary", key=f"save_edit_{q['id']}"):
                q["items"] = edited2.fillna("").to_dict(orient="records")
                st.success("Saved.")

        if st.button("Delete this quote", type="secondary", key=f"del_{q['id']}"):
            st.session_state.quotes.pop(idx)
            st.warning("Deleted. Refresh selection.")

# ---------------------------
# Tab 3: Compare
# ---------------------------
with tab3:
    st.subheader("Compare suppliers")
    if len(st.session_state.quotes) < 2:
        st.info("Add at least 2 quotes to compare.")
    else:
        vendors = [q["vendor"] for q in st.session_state.quotes]
        selected = st.multiselect("Select quotes to compare", vendors, default=vendors[:2], key="compare_select")
        sel_quotes = [q for q in st.session_state.quotes if q["vendor"] in selected]

        if len(sel_quotes) < 2:
            st.warning("Select at least 2 quotes.")
        else:
            comp = build_comparison(sel_quotes)
            st.dataframe(comp, use_container_width=True)

            flags = add_flags(comp, [q["vendor"] for q in sel_quotes])
            if flags:
                st.markdown("**Flags**")
                st.dataframe(pd.DataFrame(flags, columns=["Item","Vendor","Flag"]), use_container_width=True)
            else:
                st.success("No flags detected with current rules.")

# ---------------------------
# Tab 4: Generate PO
# ---------------------------
with tab4:
    st.subheader("Generate a Purchase Order (PO)")
    if not st.session_state.quotes:
        st.info("Add at least one quote.")
    else:
        vendors = [q["vendor"] for q in st.session_state.quotes]
        vendor = st.selectbox("Select vendor to issue PO to", vendors, index=0, key="po_vendor_select")
        q = next(q for q in st.session_state.quotes if q["vendor"] == vendor)

        po_no = st.text_input("PO Number", value=f"PO-{datetime.date.today().strftime('%Y%m%d')}-{q['id']}", key="po_number")
        currency = q.get("currency","AED")

        df = pd.DataFrame(q["items"])
        if df.empty:
            st.warning("No items to include.")
        else:
            edited = st.data_editor(df, use_container_width=True, num_rows="dynamic", key=f"po_table_{q['id']}")
            edited = edited[edited["description"].astype(str).str.strip() != ""]
            if st.button("Generate PO PDF", type="primary", key="gen_po"):
                items = edited.fillna("").to_dict(orient="records")
                for it in items:
                    it["qty"] = safe_float(it.get("qty"))
                    it["unit_price"] = safe_float(it.get("unit_price"))
                    it["total"] = safe_float(it.get("total"))
                pdf_buf = generate_po_pdf(st.session_state.company, vendor, po_no, items, currency=currency)
                st.download_button("Download PO PDF", data=pdf_buf.getvalue(),
                                   file_name=f"{po_no}.pdf", mime="application/pdf", key="dl_po")

        payload = {"exported_at": now_iso(), "company": st.session_state.company, "quotes": st.session_state.quotes}
        st.download_button("Download JSON", data=json.dumps(payload, indent=2).encode("utf-8"),
                           file_name="quote_to_po_demo_export.json", mime="application/json", key="dl_json")

st.caption("v2.1: fixed duplicate widget IDs + improved Excel header-row handling + reduced noisy rows.")
