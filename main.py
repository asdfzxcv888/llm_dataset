

import os
os.environ["HF_HOME"] = r"H:\env\hf_cache"          # main Hugging Face cache
os.environ["TRANSFORMERS_CACHE"] = r"H:\env\hf_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = r"H:\env\hf_cache"
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image
import io
import pandas as pd

from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

# ---------- CONFIG ----------
PDF_DIR = Path(r"./")  # folder containing your 1-introduction.pdf, etc.
OUT_CSV = Path("course_corpus_with_images.csv")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- LOAD CAPTION MODEL ----------
print("Loading image captioning model (BLIP)...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(DEVICE)
blip_model.eval()


def caption_image(pix):
    """
    pix: PyMuPDF Pixmap -> caption string
    """
    img_bytes = pix.tobytes("png")
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    inputs = processor(img, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = blip_model.generate(**inputs, max_new_tokens=40)

    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption


def extract_pdf_pages(pdf_path: Path):
    """
    For a single PDF, yields dicts:
    {
      "file": pdf_path.name,
      "page": page_index (0-based),
      "text": "full page text + image captions"
    }
    """
    print(f"Processing {pdf_path.name} ...")
    doc = fitz.open(pdf_path)

    for page_index, page in enumerate(doc):
        page_text = page.get_text("text") or ""
        page_text = page_text.strip()

        # Collect image captions for that page
        image_captions = []
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)

            # skip very small images (icons, etc.), optional
            if pix.width < 32 or pix.height < 32:
                continue

            try:
                cap = caption_image(pix)
                image_captions.append(f"[Image {img_index+1}: {cap}]")
            except Exception as e:
                print(f"  (warning) caption failed on {pdf_path.name} p{page_index+1}: {e}")

        # Append image captions to the end of the page text
        if image_captions:
            page_text = (
                page_text
                + "\n\n"
                + "\n".join(image_captions)
            )

        # If page is completely empty AND no images, skip
        if not page_text.strip():
            continue

        yield {
            "file": pdf_path.name,
            "page": page_index,  # zero-based; you can add +1 later if you want
            "text": page_text
        }


def build_corpus(pdf_dir: Path):
    rows = []
    for pdf in sorted(pdf_dir.glob("*.pdf")):
        # only pick your lecture + book PDFs
        if pdf.name.lower().endswith(".pdf"):
            for row in extract_pdf_pages(pdf):
                rows.append(row)
    return rows


if __name__ == "__main__":
    rows = build_corpus(PDF_DIR)
    print(f"Total pages collected: {len(rows)}")

    # Add simple row id
    for i, r in enumerate(rows):
        r["id"] = i

    df = pd.DataFrame(rows, columns=["id", "file", "page", "text"])
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved corpus to {OUT_CSV.resolve()}")
