import os
import glob
import re
from pathlib import Path
import pdfplumber

BASE_DIR = Path(__file__).parent
PDF_DIR  = BASE_DIR / 'atas_copom'
OUT_DIR  = BASE_DIR / 'atas_copom' / 'txt_copom'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Detecta início de parágrafo numerado: "1. Texto"
start_num_re = re.compile(r"^\s*(\d+)\.\s*(.+)$")
# (Opcional) cabeçalhos de seção “A)”, “B)”
sec_re       = re.compile(r"^([A-Z])\)\s*(.+)$")

for path in glob.glob(os.path.join(PDF_DIR, "*.pdf")):
    name = os.path.splitext(os.path.basename(path))[0]
    with pdfplumber.open(path) as pdf:
        raw = "\n".join(page.extract_text() or "" for page in pdf.pages)
    lines = re.sub(r"\r\n?", "\n", raw).splitlines()
    nums = [i for i, l in enumerate(lines) if start_num_re.match(l)]
    if not nums:
        print(f"⚠️  {name}: nenhum parágrafo numerado detectado")
        continue
    start_idx = nums[0]

    # 3) percorre **até o fim** a partir desse índice
    block = lines[start_idx:]

    # 4) agrupa linhas por parágrafo (com buffer), captura seções se existirem
    sections = {}
    current_sec = None
    current_num = None
    buffer = []

    for line in block:
        # checa novo cabeçalho de seção
        m_sec = sec_re.match(line)
        if m_sec:
            # antes de mudar de seção, libera o buffer atual
            if current_num is not None:
                full_text = " ".join(buffer).strip()
                sections.setdefault(current_sec or "_", []).append((current_num, full_text))
                current_num = None
                buffer = []
            current_sec = m_sec.group(1)
            sections.setdefault(current_sec, [])
            continue

        # checa início de parágrafo numerado
        m_num = start_num_re.match(line)
        if m_num:
            # salva o parágrafo anterior
            if current_num is not None:
                full_text = " ".join(buffer).strip()
                sections.setdefault(current_sec or "_", []).append((current_num, full_text))
            # inicia novo buffer
            current_num = int(m_num.group(1))
            buffer = [m_num.group(2).strip()]
        else:
            # linha de continuação
            if current_num is not None and line.strip():
                buffer.append(line.strip())

    # 5) ao final, salva o último parágrafo
    if current_num is not None:
        full_text = " ".join(buffer).strip()
        sections.setdefault(current_sec or "_", []).append((current_num, full_text))

    # 6) grava saída
    out_path = os.path.join(OUT_DIR, f"{name}_parsed.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        for sec, paras in sections.items():
            header = f"{sec})" if sec != "_" else ""
            if header:
                f.write(header + "\n")
            for num, txt in paras:
                f.write(f"{num}. {txt}\n")
            f.write("\n")

    print(f"✔️  {name} → {out_path}")
