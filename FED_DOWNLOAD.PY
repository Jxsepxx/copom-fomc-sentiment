import os
import requests
from pathlib import Path
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# URL da página de calendários e atas do FOMC
base_url = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"

# Diretório para salvar os arquivos PDF
output_dir = Path(__file__).parent / 'atas_fomc'
output_dir.mkdir(exist_ok=True)

# Cabeçalhos para a requisição HTTP
headers = {
    "User-Agent": "Mozilla/5.0"
}

# Função para baixar os arquivos PDF
def download_fomc_pdfs():
    response = requests.get(base_url, headers=headers)
    if response.status_code != 200:
        print(f"Erro ao acessar {base_url}")
        return

    soup = BeautifulSoup(response.content, 'html.parser')
    # Encontrar todos os links que apontam para arquivos PDF das atas
    pdf_links = soup.find_all('a', href=lambda href: href and 'fomcminutes' in href and href.endswith('.pdf'))

    for link in pdf_links:
        pdf_url = urljoin(base_url, link['href'])
        file_name = os.path.basename(pdf_url)
        file_path = os.path.join(output_dir, file_name)

        # Verificar se o arquivo já foi baixado
        if os.path.exists(file_path):
            print(f"Arquivo já existe: {file_name}")
            continue

        # Baixar o arquivo PDF
        print(f"Baixando: {file_name}")
        pdf_response = requests.get(pdf_url, headers=headers)
        if pdf_response.status_code == 200:
            with open(file_path, 'wb') as f:
                f.write(pdf_response.content)
            print(f"Salvo em: {file_path}")
        else:
            print(f"Erro ao baixar {pdf_url}")

# Executar a função
download_fomc_pdfs()
