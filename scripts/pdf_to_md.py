import os
import subprocess
import sys
from process_pdf import process_paper

def pdf_to_md(pdf_path):
    if not os.path.exists(pdf_path):
        print(f"Error: {pdf_path} not found")
        return

    # Extract paper name from pdf_path
    paper_name = os.path.splitext(os.path.basename(pdf_path))[0]
    base_dir = f"papers/extracted/{paper_name}"
    image_dir = os.path.join(base_dir, "images")
    text_file = os.path.join(base_dir, "text.txt")

    # Create directories
    os.makedirs(image_dir, exist_ok=True)

    print(f"Extracting text from {pdf_path}...")
    # Using -layout to preserve some formatting
    subprocess.run(["pdftotext", "-layout", pdf_path, text_file], check=True)

    print(f"Extracting images from {pdf_path}...")
    # Using -p to include page numbers and -png for format
    # Prefix "img" to match process_pdf.py's regex
    image_root = os.path.join(image_dir, "img")
    subprocess.run(["pdfimages", "-p", "-png", pdf_path, image_root], check=True)

    print(f"Converting to markdown...")
    process_paper(paper_name)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/pdf_to_md.py <path_to_pdf>")
    else:
        pdf_to_md(sys.argv[1])
