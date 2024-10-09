import os
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image
import io

# Set the size limit (in bytes) beyond which compression should be applied
SIZE_LIMIT = 20 * 1024 * 1024  # 10 MB

# Define a function to compress the PDF without using Ghostscript
def compress_pdf(input_pdf_path, output_pdf_path, image_quality=30):
    doc = fitz.open(input_pdf_path)
    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images(full=True)
        for img in image_list:
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]

            # Open the image with Pillow
            try:
                image = Image.open(io.BytesIO(image_bytes))

                # Convert images to RGB if they are not
                if image.mode in ("P", "RGBA"):
                    image = image.convert("RGB")

                # Recompress the image
                with io.BytesIO() as image_io:
                    image.save(image_io, format="JPEG", quality=image_quality)
                    image_data = image_io.getvalue()

                # Replace the image in the PDF
                page.update_image(xref, stream=image_data)
            except Exception as e:
                print(f"Could not process image {xref}: {e}")

    # Save the compressed PDF with garbage collection to reduce size
    doc.save(output_pdf_path, garbage=4)
    doc.close()

# Path to the directory containing the PDFs
pdf_dir = Path('./1536_modified')
compressed_dir = Path('./1536_compressed')
compressed_dir.mkdir(exist_ok=True)

# Loop through each PDF in the directory
for pdf_file in pdf_dir.glob("*.pdf"):
    pdf_size = pdf_file.stat().st_size

    # Check if the PDF exceeds the size limit
    if pdf_size > SIZE_LIMIT:
        output_path = compressed_dir / pdf_file.name
        print(f"Compressing {pdf_file.name} (size: {pdf_size / (960*960):.2f} MB)")

        # Compress the PDF
        compress_pdf(pdf_file, output_path)

        # Verify the compression
        compressed_size = output_path.stat().st_size
        print(f"Compressed size: {compressed_size / (960*960):.2f} MB")
    else:
        # If below the threshold, just copy it to the compressed directory
        output_path = compressed_dir / pdf_file.name
        pdf_file.replace(output_path)
