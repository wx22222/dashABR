import PyPDF2
import sys

def extract_pdf_text(pdf_path, output_path):
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        text = ''
        for i in range(len(reader.pages)):
            page = reader.pages[i]
            text += page.extract_text() + '\n'
            
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)

if __name__ == '__main__':
    extract_pdf_text(sys.argv[1], sys.argv[2])
