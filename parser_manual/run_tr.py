import os
from pathlib import Path
from PIL import Image

def main():
    # Set the PATH environment variable to include the directory where Tesseract is installed
    os.environ['PATH'] = str(Path.home() / "tesseract/bin") + os.pathsep + os.environ['PATH']
    #os.environ['TESSDATA_PREFIX'] = str(Path.home() / "tesseract/share/")
    #os.environ['PATH'] = '/home/siebenschuh/Projects/dataprep/parser_manual/tesseract/tesseract-5.3.0/.libs/libtesseract.so.5'
    os.environ['LD_LIBRARY_PATH'] = str(Path.home() / "tesseract/lib") + os.pathsep + os.environ.get('LD_LIBRARY_PATH', '')

    # Ensure pytesseract uses the correct Tesseract executable
    import pytesseract
    
    #pytesseract.pytesseract.tesseract_cmd = str(Path.home() / "tesseract/bin/tesseract")
    image = Image.open('125.png')
    text = pytesseract.image_to_string(image)


if __name__=='__main__':
    main()
