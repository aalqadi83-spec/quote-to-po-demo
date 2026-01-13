
# Quote-to-PO OS — Demo v2 (Streamlit, with OCR)

This is an interactive demo you can deploy as a **live website** (Streamlit Community Cloud) and open via a link.

## Features
- Upload quote files: **PDF / Image / XLSX / CSV**
- Optional OCR for scanned PDFs and images (English + Arabic/English)
- Review/edit line items (human-in-the-loop)
- Compare multiple quotes and show flags
- Generate a PO PDF

---

# Option A: Deploy to Streamlit Community Cloud (fastest, gives you a URL)

## 1) Create a GitHub repo
- Create a new GitHub repo named: `quote-to-po-demo`
- Upload all files in this folder (keep paths as-is)

## 2) Deploy on Streamlit Cloud
- Go to Streamlit Community Cloud
- Click **New app**
- Select your repo and set:
  - Branch: `main`
  - Main file path: `app.py`
- Click **Deploy**

Streamlit will give you a public URL you can open directly.

### Notes for OCR on Streamlit Cloud
This repo includes a `packages.txt` so Streamlit can install:
- tesseract-ocr
- tesseract Arabic language pack
- poppler utils (for PDF → images)

---

# Option B: Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

# Troubleshooting OCR
- If you see "OCR dependencies missing", make sure:
  - `tesseract` is installed
  - `poppler` is installed (for PDFs)
- Streamlit Cloud: keep `packages.txt` in the repo root.
