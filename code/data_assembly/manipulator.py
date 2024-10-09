import os
from pathlib import Path
import pandas as pd
import numpy as np
import fitz  # PyMuPDF
from PIL import Image, ImageFilter, ImageEnhance
import io
from multiprocessing import Pool

# Ensure the output directory exists
os.makedirs('./1536_modified', exist_ok=True)

# Define the root directory for original PDFs
p_root = Path('./1536_original')

# Import df
p_df = Path('/lus/eagle/projects/argonne_tpc/siebenschuh/aurora_gpt/database/scaling_data/frames/df_mod_10240.csv')
df_10240 = pd.read_csv(p_df, sep='|')

# Filter the manipulated PDFs
df_manip = df_10240[df_10240['manipulated'] == 1]

# Extract file names
df_manip['file_name'] = df_manip['path'].str.split('/').str[-1]

try:
    from PIL import ImageResampling
    resample_filter = ImageResampling.LANCZOS
except ImportError:
    resample_filter = Image.LANCZOS

def process_pdf(row):
    """Process a single PDF file based on a row from df_manip."""
    corruption_dict_list = []  # Initialize a list to store corruption parameters for each page

    src_pdf_path = Path(p_root) / str(row['path']).split('/')[-1]
    text_to_be_inserted = row['text']
    src_file_name = row['file_name']
    dst_pdf_path = os.path.join('./1536_modified', src_file_name)
    
    # Open the source PDF
    doc = fitz.open(src_pdf_path)
    new_doc = fitz.open()  # New PDF for modified pages
    num_pages = len(doc)
    
    # Divide the text among pages
    text_length = np.random.randint(2000, 3001)
    text_to_add = str(row['text'])[:text_length]
    chars_per_page = max(1, text_length // num_pages)
    text_pages = [
        text_to_add[i * chars_per_page:(i + 1) * chars_per_page]
        for i in range(num_pages)
    ]
    remaining_chars = text_length - (chars_per_page * num_pages)
    if remaining_chars > 0:
        text_pages[-1] += text_to_add[-remaining_chars:]

    for i in range(num_pages):
        page = doc.load_page(i)
        
        # Render page to image with reduced zoom to lower resolution
        zoom = 1.4  # Reduced from 2 to 1
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes('png')
        img = Image.open(io.BytesIO(img_data))
        
        # Apply random transformations
        blur_radius = np.random.uniform(0, 1)
        contrast_factor = np.random.uniform(0.85, 1.15)
        rotation_angle = np.random.uniform(-0.8, 0.8)

        blurred_img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        enhancer = ImageEnhance.Contrast(blurred_img)
        enhanced_img = enhancer.enhance(contrast_factor)
        rotated_img = enhanced_img.rotate(rotation_angle, expand=True)
        
        # Optionally resize the image to reduce resolution further
        max_width, max_height = 960, 1280  # Adjust as needed
        rotated_img.thumbnail((max_width, max_height), resample=resample_filter)
        
        # Save transformation details
        corr_dict_page = {
            'file_name': src_file_name,
            'page_number': i + 1,
            'blur_radius': blur_radius,
            'contrast_factor': contrast_factor,
            'rotation_angle': rotation_angle
        }
        corruption_dict_list.append(corr_dict_page)
        
        # Convert the PIL image back to bytes
        img_byte_arr = io.BytesIO()
        # Save as JPEG with reduced quality
        quality_value = 60  # Adjust quality between 1 (worst) and 95 (best)
        rotated_img = rotated_img.convert('RGB')  # Ensure image is in RGB mode
        rotated_img.save(img_byte_arr, format='JPEG', quality=quality_value)
        img_byte_arr = img_byte_arr.getvalue()
        
        # Create a new PDF page with the image
        # Adjust the page size to match the image size
        img_width, img_height = rotated_img.size
        img_rect = fitz.Rect(0, 0, img_width, img_height)
        new_page = new_doc.new_page(width=img_width, height=img_height)
        new_page.insert_image(img_rect, stream=img_byte_arr)
        
        # Add invisible text to the page
        text_chunk = text_pages[i]
        transparent_color = (1, 1, 1, 0)  # Fully transparent color
        new_page.insert_text(
            point=(50, 50),
            text=text_chunk,
            fontsize=0.1,  # Super small font size
            fontname="helv",
            color=transparent_color
        )
    
    # Save the new PDF with optimization
    new_doc.save(dst_pdf_path, garbage=4, deflate=True)
    new_doc.close()
    doc.close()
    
    print(f'Done processing {src_file_name}')
    return corruption_dict_list  # Return the corruption parameters for each page

# Run the processing in parallel
if __name__ == "__main__":
    with Pool(10) as pool:  # Create a pool of 6 workers
        corruption_results = pool.map(process_pdf, [row for _, row in df_manip.iterrows()])

    # Flatten the list of corruption results and save to CSV
    corruption_dict_list = [item for sublist in corruption_results for item in sublist]
    corruption_df = pd.DataFrame(corruption_dict_list)

    # store
    corruption_df.to_csv('corruption_log.csv', index=False)
