import os
import re

def process_paper(paper_name):
    base_dir = f"papers/extracted/{paper_name}"
    text_file = os.path.join(base_dir, "text.txt")
    output_md = os.path.join(base_dir, "paper.md")
    image_dir = "images"
    
    if not os.path.exists(text_file):
        print(f"Error: {text_file} not found")
        return

    with open(text_file, "r") as f:
        content = f.read()

    # Split by page (form feed character \x0c)
    pages = content.split('\x0c')
    
    # Get list of images
    images = sorted(os.listdir(os.path.join(base_dir, image_dir)))
    
    # Map images to pages
    page_to_images = {}
    for img in images:
        match = re.search(r'img-(\d+)-\d+', img)
        if match:
            page_num = int(match.group(1))
            if page_num not in page_to_images:
                page_to_images[page_num] = []
            page_to_images[page_num].append(img)

    md_content = f"# {paper_name.replace('_', ' ').title()}\n\n"
    
    for i, page in enumerate(pages):
        page_num = i + 1
        # Basic cleanup: remove headers/footers (approximate)
        # Assuming page numbers or titles at top/bottom
        lines = page.strip().split('\n')
        if not lines:
            continue
            
        # Add page text
        md_content += f"## Page {page_num}\n\n"
        md_content += page.strip() + "\n\n"
        
        # Insert images for this page
        if page_num in page_to_images:
            for img in page_to_images[page_num]:
                md_content += f"![Image Page {page_num}]({image_dir}/{img})\n\n"

    with open(output_md, "w") as f:
        f.write(md_content)
    
    print(f"Generated {output_md}")

if __name__ == "__main__":
    process_paper("masked_autoencoder")
