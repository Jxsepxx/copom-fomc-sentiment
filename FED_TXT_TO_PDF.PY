import os
import glob
import re
from pathlib import Path
import pdfplumber

# Diretório dos PDFs das atas do FOMC
BASE_DIR = Path(__file__).parent
PDF_DIR  = BASE_DIR / 'atas_fomc' / 'atas_pdf'
OUT_DIR  = BASE_DIR / 'atas_fomc' / 'atas_txt'
OUT_DIR.mkdir(parents=True, exist_ok=True)

def clean_text_fomc(text):
    lines = text.split('\n')
    clean_lines = []

    for line in lines:
        line = line.strip()

        # Ignora linhas vazias ou muito curtas
        if not line or len(line.split()) <= 3:
            continue

        # Remove páginas, nomes de rodapé e URLs
        if re.search(r'Page \d+|Federal Open Market Committee', line):
            continue
        if 'https://' in line or 'www.' in line:
            continue
        if re.match(r'^[A-Z][a-z]+\s+[A-Z][a-z]+,', line):  # nomes com vírgula
            continue
        if re.match(r'^[\d\w\s]*Board of Governors', line):
            continue

        # Junta frases quebradas
        line = re.sub(r'\s+', ' ', line)
        clean_lines.append(line)

    full_text = ' '.join(clean_lines)
    full_text = re.sub(r'\s+([.,;!?])', r'\1', full_text)
    full_text = re.sub(r'\s{2,}', ' ', full_text)

    return full_text.strip()


for path in glob.glob(os.path.join(PDF_DIR, "*.pdf")):
    name = os.path.splitext(os.path.basename(path))[0]

    with pdfplumber.open(path) as pdf:
        raw_text = "\n".join(page.extract_text() or "" for page in pdf.pages)

    clean = clean_text_fomc(raw_text)

    out_path = os.path.join(OUT_DIR, f"{name}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(clean)

    print(f"✔️ {name} salvo em: {out_path}")
