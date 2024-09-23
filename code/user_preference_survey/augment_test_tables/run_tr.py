from PIL import Image
import pytesseract
import pymupdf
from PIL import Image
import io
import os
import sys
from pathlib import Path
import random

# add the parent directory to sys.path
sys.path.append(str(Path('..').resolve()))
from table_utils import get_frames_of_choices_raw


def main():
    # imports
    os.environ['LD_LIBRARY_PATH'] = str(Path.home() / "tesseract/lib")
    pytesseract.pytesseract.tesseract_cmd = str(Path.home() / "tesseract/bin/tesseract") # for teserract


    # get test
    _, df_test, _ = get_frames_of_choices_raw()
    
    p_root=Path('/lus/eagle/projects/argonne_tpc/siebenschuh/aurora_gpt/joint')
    p_img_dst = Path('./images')

    # PDFs
    pdf_path_list = list(df_test['path'])
    pdf_page_list = list(df_test['page'])
    html_groundtruth_list = list(df_test['html'])
    
    # tesseract OCR
    image_file_name_list = []
    tesseract_list = []
    dpi_sizes = []
    
    # loop elements
    for pdf_file_name, page_idx, html_text in zip(pdf_path_list, pdf_page_list, html_groundtruth_list):
        # path
        pdf_path = p_root / pdf_file_name
        if pdf_path.is_file():
            doc = pymupdf.open(pdf_path) # open the PDF
            rnd_dpi = random.choice([60, 100])
            
            pixmap = doc[page_idx].get_pixmap(dpi=rnd_dpi)
            page_img = pixmap.tobytes()
            image = Image.open(io.BytesIO(page_img))
    
            # image name
            img_path = str(p_img_dst / f"{Path(pdf_file_name).stem}_{page_idx}.png")
            
            # store
            image.save(img_path)
    
            # load tesseract image
            text = pytesseract.image_to_string(image)
            tesseract_list.append(text)
    
            # append lists
            dpi_sizes.append(rnd_dpi)
            image_file_name_list.append(img_path)
    
            break

# entry
if __name__=='__main__':
    main()