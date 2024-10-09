import os
import pandas as pd
import json
from pathlib import Path
import pymupdf
import numpy as np
import random
import string
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import fitz  # PyMuPDF library
import io

def zip_files(file_paths, output_zip):
    with zipfile.ZipFile(output_zip, 'w') as zipf:
        for file in file_paths:
            # Add each file to the zip archive
            zipf.write(file, os.path.basename(file))


def mode(series):
    return series.mode()[0]  # Get the first mode in case of ties

def remove_text_layer(src_path: Path, dst_path: Path) -> None:
    """
    Removes the text layer from a PDF by rasterizing each page and saving it back.
    Converts the PDF pages to images and saves as a new PDF without selectable text.
    """
    src_path = Path(src_path)
    dst_path = Path(dst_path)

    # Load the source PDF
    doc = pymupdf.open(src_path)
    
    # Create a new PDF for saving
    new_doc = pymupdf.open()  # Create a new, empty PDF document
    
    # Iterate through pages and render them as images
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        
        # Convert each page to a high-resolution image
        pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2))  # Adjust resolution (2, 2) for better quality
        
        # Create a new page in the new PDF and insert the image
        img_page = new_doc.new_page(width=pix.width, height=pix.height)
        img_page.insert_image(img_page.rect, pixmap=pix)
    
    # Save the new PDF without a text layer
    new_doc.save(dst_path)
    new_doc.close()
    doc.close()

def simulated_scanned_effect(src_path: Path, dst_path: Path) -> dict:
    """
    Convert a digital-born PDF to look as if it were scanned by adding random imperfections
    - noise
    - blur 
    - rotation
    - brightness
    - adjustments.
    """
    src_path = Path(src_path)
    dst_path = Path(dst_path)

    # Check for input PDF file validity
    assert src_path.is_file() and str(src_path).endswith('.pdf'), f"Input file must exist and be PDF but `src_path` does not: {src_path}" 

    # Load the PDF
    doc = fitz.open(src_path)
    
    # Create a new PDF for saving the manipulated pages
    new_doc = fitz.open()  # Empty document

    # Hyperparameters (of imperfections)
    rot_list = []
    blur_rad_list = []
    brightness_list = []
    contrast_list = []
    rnd_noise_list = []
    
    # Iterate over each page in the PDF
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        
        # Convert page to an image (pixmap)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # Higher resolution
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # Apply imperfections to simulate a scanned document
        # - slight rotation
        rndRot = random.uniform(-2, 2)
        img = img.rotate(rndRot, expand=True)
        
        # - add Gaussian blur
        rndBlurRadius = random.uniform(1, 2)
        img = img.filter(ImageFilter.GaussianBlur(radius=rndBlurRadius))
        
        # - adjust brightness and contrast
        rndBright = random.uniform(0.9, 1.1)
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(rndBright)
        
        rndContrast = random.uniform(0.9, 1.1)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(rndContrast)

        # Add noise
        img = img.convert("L")  # Convert to grayscale to add noise
        rndNoise = random.uniform(10, 20)
        noise = Image.effect_noise(img.size, rndNoise)
        img = Image.blend(img, noise, 0.2)  # Blend the noise into the image
        img = img.convert("RGB")  # Convert back to RGB

        # Convert the PIL image to a Pixmap through an in-memory buffer
        img_buffer = io.BytesIO()
        img.save(img_buffer, format="PNG")
        img_buffer.seek(0)
        pixmap = fitz.Pixmap(img_buffer)  # Create Pixmap from buffer

        # Create a new page in the output PDF and insert the image
        img_page = new_doc.new_page(width=pixmap.width, height=pixmap.height)
        img_page.insert_image(img_page.rect, pixmap=pixmap)

        # Append transformation values
        rot_list.append(rndRot)
        blur_rad_list.append(rndBlurRadius)
        brightness_list.append(rndBright)
        contrast_list.append(rndContrast)
        rnd_noise_list.append(rndNoise)

    # Compute mean transformation values
    hyperpara_dict = {
        'mean_rotation': np.mean(rot_list), 
        'mean_blur_radius': np.mean(blur_rad_list),
        'mean_brightness': np.mean(brightness_list),
        'mean_contrast': np.mean(contrast_list),
        'mean_noise': np.mean(rnd_noise_list),
    }
        
    # Save the new manipulated PDF
    new_doc.save(src_path, incremental=True)
    new_doc.close()
    doc.close()

    return hyperpara_dict



# Substitute random characters in a string
def random_character_substitution(s, p=0.005):
    """
    Change 0.5% of characters
    """
    return ''.join(
        random.choice(string.ascii_letters) if random.random() < p else c 
        for c in s
    )

# Delete random characters in a string
def random_character_deletion(s, p=0.005):
    """
    Delete 0.5% of characters
    """
    return ''.join(c for c in s if random.random() > p)

# Insert random spaces within words
def insert_random_spaces(s, p=0.01):
    """
    Insert random whitespaces
    """
    return ''.join(
        c + ' ' if random.random() < p and c.isalpha() else c 
        for c in s
    )

# Merge random words together by removing spaces
def merge_random_words(s, p=0.02):
    """
    Merge 2% of all word pairs
    """
    words = s.split()
    for i in range(len(words) - 1):
        if random.random() < p:
            words[i] += words.pop(i + 1)
    return ' '.join(words)

# OCR-like character substitutions
def ocr_like_substitutions(s, p=0.05):
    """
    Introduce classical char error
    """
    ocr_errors = {'O': '0', 'I': '1', 'S': '5', 'B': '8', 'G': '6', 'Z': '2'}
    return ''.join(
        ocr_errors.get(c, c) if random.random() < p and c in ocr_errors else c 
        for c in s
    )

def text_scrambler(s, 
                   subst_p=0.005, 
                   del_p=0.005, 
                   space_p=0.01, 
                   merge_p=0.02, 
                   ocr_p=0.05):
    """
    Text scrambler function
    """
    s = random_character_substitution(s, p=subst_p)
    s = random_character_deletion(s, p=del_p)
    s = insert_random_spaces(s, p=space_p)
    s = merge_random_words(s, p=merge_p)
    s = ocr_like_substitutions(s, p=ocr_p)
    return s