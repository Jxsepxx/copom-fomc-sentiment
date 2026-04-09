# -*- coding: utf-8 -*-
# FOMC-METER Script Completo com Testes de Overfitting (Correlação Train vs Test)

import os, re
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sentence_transformers import SentenceTransformer
from scipy.stats import zscore, pearsonr
from statsmodels.tsa.filters.hp_filter import hpfilter
import matplotlib.dates as mdates

nltk.download('punkt')

# -------------------------
# Diretórios e caminhos
# -------------------------
INPUT_DIR     = BASE_DIR / 'atas_fomc' / 'atas_txt'
FED_RATE_PATH = BASE_DIR / 'FEDFUNDS.xlsx'
GUIDANCE_PATH = BASE_DIR / 'guidance_rotulado_200_corrigido.xlsx'
OUTPUT_DIR    = BASE_DIR / 'outputs'

# Palavras-chave para filtrar parágrafos relevantes
KEYWORDS = ['monetary', 'interest', 'inflation', 'policy', 'rate', 'funds']

# ------------------------------------------------
# 1) Carregar e treinar classificador de guidance
# ------------------------------------------------
df_guid = pd.read_excel(GUIDANCE_PATH)
df_guid = df_guid.dropna(subset=['texto', 'tipo'])
df_guid['tipo'] = df_guid['tipo'].str.lower().str.strip()
df_guid = df_guid[df_guid['tipo'].isin(['hawkish', 'dovish', 'neutral'])]

# Usar modelo multilingue para embeddings
model_st = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
emb_guid = model_st.encode(df_guid['texto'].tolist(), convert_to_tensor=False)

Xg_tr, Xg_te, yg_tr, yg_te = train_test_split(
    emb_guid, df_guid['tipo'], test_size=0.2, random_state=42, stratify=df_guid['tipo']
)
clf_guid = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,            # impede árvores muito profundas
    min_samples_leaf=5,     # cada folha deve ter pelo menos 5 exemplos
    max_features='sqrt',    # default, mas garanta que não seja “auto” (todas as features)
    random_state=42,
    n_jobs=-1
)
clf_guid.fit(Xg_tr, yg_tr)

def classify_guidance(text: str) -> int:
    emb = model_st.encode([text], convert_to_tensor=False)
    pred = clf_guid.predict(emb)[0]
    return {'hawkish': 1, 'neutral': 0, 'dovish': -1}[pred]

# Avaliar acurácia in-sample e out-of-sample do classificador de guidance
yg_tr_pred = clf_guid.predict(Xg_tr)
yg_te_pred = clf_guid.predict(Xg_te)
acc_tr_guid = (yg_tr_pred == yg_tr).mean()
acc_te_guid = (yg_te_pred == yg_te).mean()
print(f"Acurácia Guidance (Treino): {acc_tr_guid:.3f}")
print(f"Acurácia Guidance (Teste): {acc_te_guid:.3f}")

# ------------------------------------------------
# 2) Processar atas do FOMC e calcular raw_score
# ------------------------------------------------
pattern = re.compile(r'fomcminutes(\d{8})', flags=re.IGNORECASE)
rows, seen = [], set()

for fn in sorted(os.listdir(INPUT_DIR)):
    if not fn.lower().endswith('.txt'):
        continue
    m = pattern.search(fn)
    if not m:
        continue

    date = pd.to_datetime(m.group(1), format='%Y%m%d')
    path = os.path.join(INPUT_DIR, fn)
    paras = [line.strip() for line in open(path, encoding='utf-8') if line.strip()]
    clean = [p for p in paras if p not in seen]
    seen = set(paras)

    score, count = 0.0, 0
    for p in clean:
        text = p.lower()
        if any(kw in text for kw in KEYWORDS):
            score += classify_guidance(p)
        count += 1
    count = max(count, 3)
    rows.append({'date': date, 'guidance_score': score / count})

fomc = pd.DataFrame(rows).sort_values('date')

# ------------------------------------------------
# 3) Suavização e normalização do indicador bruto
# ------------------------------------------------
# Rolling mean de 3 atas (suavização leve)
fomc['score_smooth_raw'] = fomc['guidance_score'].rolling(3, center=True, min_periods=1).mean()

# Normalizar em [-1, +1]
fomc['score_smooth'] = MinMaxScaler(feature_range=(-1, 1)).fit_transform(
    fomc[['score_smooth_raw']].fillna(0)
)

# ------------------------------------------------
# 4) Carregar e escalar série Fed Funds Rate
# ------------------------------------------------
df_fed = pd.read_excel(FED_RATE_PATH)
df_fed['date'] = pd.to_datetime(df_fed['date'], errors='coerce')
df_fed = df_fed[(df_fed['date'] >= '2007-10-31') & (df_fed['date'] <= '2025-12-31')]
df_fed['delta_t1'] = df_fed['FEDFUNDS'].shift(-1)
df_fed['fed_scaled'] = MinMaxScaler(feature_range=(-1, 1)).fit_transform(
    df_fed[['delta_t1']].fillna(0)
)
df_fed = df_fed[['date', 'fed_scaled']]

# ------------------------------------------------
# 5) Merge das séries (atas + Fed Funds)
# ------------------------------------------------
base = pd.merge_asof(
    fomc[['date', 'score_smooth']].sort_values('date'),
    df_fed.sort_values('date'),
    on='date', direction='backward'
)
base.dropna(inplace=True)

# ------------------------------------------------
# 6) Calcular surpresa usando HP filter
# ------------------------------------------------
cop = base['score_smooth'].values
trend = np.zeros_like(cop)
cycle = np.zeros_like(cop)

for i in range(len(cop)):
    if i < 5:
        trend[i] = cop[i]
        cycle[i] = 0
    else:
        t_i, c_i = hpfilter(cop[:i + 1], lamb=400)
        trend[i] = t_i[-1]
        cycle[i] = cop[i] - t_i[-1]

base['surprise'] = cycle

# Features finais
base['surprise_lag1'] = zscore(pd.Series(base['surprise']).shift(1).fillna(0))
thr = np.nanpercentile(np.abs(base['surprise']), 90)
base['extreme_shock'] = (np.abs(base['surprise']) >= thr).astype(int)
base['interaction'] = base['surprise_lag1'] * base['extreme_shock']

# ------------------------------------------------
# 7) Separação temporal para testar overfitting
# ------------------------------------------------
# Definir data de corte: 
cutoff_date = pd.to_datetime("2020-01-01")
mask_train = base['date'] < cutoff_date
mask_test  = base['date'] >= cutoff_date

X_full = base[['surprise_lag1', 'extreme_shock', 'interaction']].values
y_full = base['fed_scaled'].values

X_train = base.loc[mask_train, ['surprise_lag1', 'extreme_shock', 'interaction']].values
y_train = base.loc[mask_train, 'fed_scaled'].values
X_test  = base.loc[mask_test, ['surprise_lag1', 'extreme_shock', 'interaction']].values
y_test  = base.loc[mask_test, 'fed_scaled'].values

print(f"Número de observações (Total / Train / Test): {len(base)}/{mask_train.sum()}/{mask_test.sum()}")

# ------------------------------------------------
# 8) Treinar RandomForest no conjunto de treino
# ------------------------------------------------
rf_train = RandomForestRegressor(n_estimators=200, random_state=42)
rf_train.fit(X_train, y_train)

# Predições in-sample (treino) e out-of-sample (teste)
y_train_pred = rf_train.predict(X_train)
y_test_pred  = rf_train.predict(X_test)

# Calcular correlações
corr_train = pearsonr(y_train_pred, y_train)[0]
corr_test  = pearsonr(y_test_pred, y_test)[0]
print(f"Correlação (Train) FOMC-METER vs ΔFed Funds: {corr_train:.3f}")
print(f"Correlação (Test)  FOMC-METER vs ΔFed Funds: {corr_test:.3f}")

# Adicionar as colunas preditas ao DataFrame
base['fomc_meter_train'] = np.nan
base.loc[mask_train, 'fomc_meter_train'] = y_train_pred
base['fomc_meter_test'] = np.nan
base.loc[mask_test,  'fomc_meter_test']  = y_test_pred

# ------------------------------------------------
# 9) Treinar RF no conjunto completo (in-sample)
# ------------------------------------------------
rf_full = RandomForestRegressor(n_estimators=200, random_state=42)
rf_full.fit(X_full, y_full)
base['fomc_meter'] = rf_full.predict(X_full)

# ------------------------------------------------
# 10) Importância das Features + residual (in-sample)
# ------------------------------------------------
imp = pd.DataFrame({
    'feature': ['surprise_lag1', 'extreme_shock', 'interaction'],
    'importance': rf_full.feature_importances_
}).sort_values('importance', ascending=False)

print("\nImportância das features (modelo completo):")
print(imp)

base['residual'] = base['fomc_meter'] - base['fed_scaled']

plt.figure(figsize=(10, 4))
plt.plot(base['date'], base['residual'], label='Residual (in-sample)')
plt.axhline(0, color='gray', linestyle='--')
plt.title('Residual FOMC-METER (in-sample) vs ΔFed Funds')
plt.grid(True)
plt.tight_layout()
plt.show()

# ------------------------------------------------
# 11) Zonas Hawkish/Dovish Dinâmicas
# ------------------------------------------------
hawk_th = base['fomc_meter'].quantile(0.75)
dove_th = base['fomc_meter'].quantile(0.25)

print(f"\nHawkish threshold (75th percentile): {hawk_th:.3f}")
print(f"Dovish threshold (25th percentile): {dove_th:.3f}")

# ------------------------------------------------
# 12) Plot FOMC-METER vs ΔFed Funds (in-sample)
# ------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(base['date'], base['fomc_meter'], label='FOMC-METER (in-sample)', color='tab:red', linewidth=2)
ax.plot(base['date'], base['fed_scaled'], label='Δ Fed Funds t+1', color='black', linewidth=1.5)

ax.axhspan(hawk_th, base['fomc_meter'].max()+0.01, facecolor='red', alpha=0.1, label='Hawkish Zone')
ax.axhspan(base['fomc_meter'].min()-0.01, dove_th, facecolor='green', alpha=0.1, label='Dovish Zone')

ax.set_title('FOMC-METER vs Δ Fed Funds (In-Sample)', fontsize=14)
ax.set_xlabel('Date'); ax.set_ylabel('Scaled Value')
ax.legend(loc='upper left'); ax.grid(ls=':', alpha=0.5)
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.tight_layout()
plt.show()

# ------------------------------------------------
# 13) Plot FOMC-METER (Train vs Test) e ΔFed Funds
# ------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 4))
# Predições no conjunto de treino
ax.plot(base.loc[mask_train, 'date'], base.loc[mask_train, 'fomc_meter_train'],
        label='FOMC-METER (Treino)', color='tab:orange', linewidth=2)
# Predições no conjunto de teste
ax.plot(base.loc[mask_test, 'date'], base.loc[mask_test, 'fomc_meter_test'],
        label='FOMC-METER (Teste)',  color='tab:blue',   linewidth=2)
# Série real
ax.plot(base['date'], base['fed_scaled'], label='Δ Fed Funds t+1', color='black', linewidth=1.5)

# Zonas com base no in-sample (mantemos mesmos thresholds)
ax.axhspan(hawk_th, base['fomc_meter'].max()+0.01, facecolor='red', alpha=0.1)
ax.axhspan(base['fomc_meter'].min()-0.01, dove_th, facecolor='green', alpha=0.1)

ax.set_title('FOMC-METER (Treino vs Teste) vs Δ Fed Funds', fontsize=14)
ax.set_xlabel('Date'); ax.set_ylabel('Scaled Value')
ax.legend(loc='upper left'); ax.grid(ls=':', alpha=0.5)
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.tight_layout()
plt.show()

# ------------------------------------------------
# 14) Análise de correlação geral
# ------------------------------------------------
corr_full = pearsonr(base['fomc_meter'], base['fed_scaled'])[0]
print(f"\nCorrelação (In-Sample) FOMC-METER vs Δ Fed Funds: {corr_full:.3f}")

# Correlação entre cada feature e o alvo
for feat in ['surprise_lag1', 'extreme_shock', 'interaction']:
    c_feat = pearsonr(base['fed_scaled'], base[feat])[0]
    print(f"Correlation ΔFed Funds vs {feat}: {c_feat:.3f}")

# ------------------------------------------------
# 15) Exportar resultados finais
# ------------------------------------------------
base.to_excel(os.path.join(OUTPUT_DIR, 'fomc_meter_output_with_oof_tests.xlsx'), index=False)
print("\nExport completed.")





'''
A seguir apresento uma explicação **detalhada**, passo a passo, de tudo o que o script faz, quais são as intenções de cada trecho e quais cuidados devemos ter (como evitar overfitting, escolher um bom pipeline de transformação, etc.). Vou dividir a explicação em seções numeradas que correspondem exatamente às etapas do código, comentando contexto, lógica e riscos para cada parte.

---

## 1) Importações e configurações gerais

```python
import os, re
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sentence_transformers import SentenceTransformer
from scipy.stats import zscore, pearsonr
from statsmodels.tsa.filters.hp_filter import hpfilter
import matplotlib.dates as mdates

nltk.download('punkt')
```

* **Bibliotecas carregadas**

  * `os` e `re` (regex): para navegar em pastas e extrair datas a partir do nome dos arquivos de ata.
  * `pandas` e `numpy`: para leitura, manipulação de tabelas e operações numéricas.
  * `matplotlib.pyplot` e `mdates`: para gerar gráficos de série temporal (eixos de data).
  * `nltk` (com `punkt`): para garantir tokenização, embora no script em si este download seja residual – não fazemos tokenização explícita de NLTK, mas mantemos por precaução (caso queiramos pré-processar texto).
  * `sklearn.preprocessing.MinMaxScaler`: para converter qualquer série em escala \[–1, +1], garantindo comparabilidade entre scores qualitativos e séries monetárias.
  * `sklearn.model_selection.train_test_split`: para separar uma parte das frases rotuladas em treino e teste, evitando usar todo o conjunto na mesma fase de treino (reduz overfitting).
  * `sklearn.ensemble.RandomForestClassifier` e `RandomForestRegressor`:

    * **`RandomForestClassifier`** vai classificar cada parágrafo (hawkish/dovish/neutral).
    * **`RandomForestRegressor`** vai “ler” as features de surpresa/choque e tentar prever o Δ Fed Funds.
  * `sentence_transformers.SentenceTransformer`: para converter cada frase curta (texto de ata ou frase rotulada) num vetor numérico (“embedding”) em alta dimensão, preservando proximidade semântica.
  * `scipy.stats.zscore` e `pearsonr`: para padronizar (z-score) e calcular correlações de Pearson, respectivamente.
  * `statsmodels.tsa.filters.hp_filter.hpfilter`: o filtro de Hodrick-Prescott, utilizado para decompor uma série em tendência + ciclo (surpresa).

* **`nltk.download('punkt')`**: baixa o pacote de tokenização que eventualmente pode ser usado. Não é essencial aqui, mas assevera que NLTK está pronto.

---

## 2) Definição de caminhos e parâmetros

```python
INPUT_DIR     = BASE_DIR / 'atas_fomc' / 'atas_txt'
FED_RATE_PATH = BASE_DIR / 'FEDFUNDS.xlsx'
GUIDANCE_PATH = BASE_DIR / 'guidance_rotulado_200_corrigido.xlsx'
OUTPUT_DIR    = BASE_DIR / 'outputs'

# Palavras-chave para filtrar parágrafos relevantes
KEYWORDS = ['monetary', 'interest', 'inflation', 'policy', 'rate', 'funds']
```

* **`INPUT_DIR`**: pasta que contém os arquivos de ata do FOMC, todos em formato `.txt`.
* **`FED_RATE_PATH`**: arquivo Excel com a série histórica de taxas do Fed (coluna “FEDFUNDS”).
* **`GUIDANCE_PATH`**: base de frases rotuladas (200 frases) em português, com colunas `texto` e `tipo` (hawkish/dovish/neutral). Essa base *já* foi traduzida ou criada em inglês, mas neste script nós carregamos diretamente em `texto` (supõe-se que esteja em inglês ou português coerente com o classificador).
* **`OUTPUT_DIR`**: pasta onde vamos salvar o resultado final (o Excel com dados do `fomc_meter`).
* **`KEYWORDS`**: lista de palavras que usamos para determinar se um parágrafo da ata é relevante para política monetária.

  * Se o parágrafo contiver pelo menos uma dessas palavras (case-insensitive), ele será enviado ao classificador de guidance.
  * Exemplo: `"monetary"`, `"interest"`, `"inflation"`, `"policy"`, `"rate"`, `"funds"`.
  * Este filtro é importante para reduzir ruído: muitas frases da ata podem tratar de aspectos administrativos, calendário ou balanço de riscos sem jamais falar diretamente de política monetária.

> **Por que `KEYWORDS`?**
>
> * Se classificarmos *todos* os parágrafos, incluímos trechos totalmente alheios (ex.: discussões sobre “compliance” ou “risks to financial stability” sem menção direta à taxa, o que dilui a pontuação).
> * Ao filtrar *apenas* parágrafos que contenham termos tipicamente associados a política monetária, o score de guidance reflete melhor o “tom” que o Fed passa quando fala de juros, inflação e cenários macro.

---

## 3) Treinamento do classificador de guidance

```python
df_guid = pd.read_excel(GUIDANCE_PATH)
df_guid = df_guid.dropna(subset=['texto', 'tipo'])
df_guid['tipo'] = df_guid['tipo'].str.lower().str.strip()
df_guid = df_guid[df_guid['tipo'].isin(['hawkish', 'dovish', 'neutral'])]
```

1. **Carregamento da base rotulada**

   * `df_guid` é um DataFrame contendo, idealmente, 200 frases com colunas:

     * **`texto`**: a frase em português (ou inglês) que descreve um trecho tipicamente extraído do Copom ou de comunicados.
     * **`tipo`**: rótulo textual, que assume valores em minúsculo: `"hawkish"`, `"neutral"` ou `"dovish"`.

2. **Limpeza inicial**

   * `dropna(subset=['texto','tipo'])`: descarta qualquer linha que não tenha `texto` ou `tipo`.
   * `df_guid['tipo'] = df_guid['tipo'].str.lower().str.strip()`: unifica formatação dos rótulos (para evitar espaços extras ou letras maiúsculas).
   * `df_guid = df_guid[df_guid['tipo'].isin(['hawkish','dovish','neutral'])]`: garante que só fiquem as 3 categorias que reconhecemos.

---

### 3.1) Geração de embeddings

```python
model_st = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
emb_guid = model_st.encode(df_guid['texto'].tolist(), convert_to_tensor=False)
```

* **`SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')`**

  * É um modelo pré-treinado que gera vetores de 384 ou 768 dimensões (dependendo da versão) para frases em múltiplos idiomas, incluindo português e inglês.
  * Cada frase em `df_guid['texto']` será transformada em um vetor numérico:

    ```python
    emb_guid = [embedding(frase1), embedding(frase2), ..., embedding(fraseN)]
    ```
  * Resultado:

    * **`emb_guid`**: uma matriz de shape `(n_frases, n_dim)` (n\_dim ≈ 384 ou 768).

> **Por que usar embeddings?**
>
> * Modelos de linguagem (como BERT, RoBERTa, MiniLM) capturam semântica: frases com sentido próximo ficam próximas no espaço vetorial.
> * Com esses vetores, podemos alimentar um algoritmo clássico (RandomForest) para aprender padrões de diferença entre “hawkish”, “dovish” e “neutral”.

---

### 3.2) Divisão treino/teste e treinamento

```python
Xg_tr, Xg_te, yg_tr, yg_te = train_test_split(
    emb_guid, df_guid['tipo'],
    test_size=0.2, random_state=42, stratify=df_guid['tipo']
)
clf_guid = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
clf_guid.fit(Xg_tr, yg_tr)
```

1. **`train_test_split`**

   * **`emb_guid`** são os vetores (features)
   * **`df_guid['tipo']`** são os rótulos (y)
   * `test_size=0.2`: 20% das frases serão reservadas para teste, 80% para treino.
   * `stratify=df_guid['tipo']`: garante que a proporção de `"hawkish"`, `"neutral"`, `"dovish"` seja mantida tanto no conjunto de treino quanto no de teste. Isso é crucial para evitar o *imbalanced split*.

2. **`RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)`**

   * **`n_estimators=200`**: O modelo vai treinar 200 árvores de decisão.
   * **`random_state=42`**: fixa a semente para garantir reprodutibilidade.
   * **`n_jobs=-1`**: utiliza todos os núcleos de CPU para paralelizar a construção das árvores.
   * **Treinamento (`fit`)**:

     * A cada árvore, escolhe-se aleatoriamente (com amostragem *bootstrap*) parte dos vetores de embeddings e parte (aleatória) das features (o `max_features` padrão é a raiz quadrada do número de dimensões).
     * Isso reduz a chance de *overfitting* de cada árvore isolada: cada árvore “vê” uma versão diferente dos dados.
     * O ensemble final (votação majoritária) tende a ter melhor generalização que uma única árvore.

> **Como evitar overfitting no classificador?**
>
> * **`max_depth`** pode ser ajustado (profundidade das árvores).
> * **`min_samples_leaf`**: número mínimo de amostras em cada folha.
> * **`n_estimators`** muito grande tende a melhorar, mas chega um ponto de *diminuição de retornos*.
> * Importante checar a acurácia tanto no treino quanto no teste (ex.: `clf_guid.score(Xg_te, yg_te)`). Se a acurácia no treino for muito alta (∼1.0) e no teste baixa (<0.7, digamos), significa overfitting.

---

## 4) Processamento das atas do FOMC

```python
pattern = re.compile(r'fomcminutes(\d{8})', flags=re.IGNORECASE)
rows, seen = [], set()

for fn in sorted(os.listdir(INPUT_DIR)):
    if not fn.lower().endswith('.txt'): 
        continue
    m = pattern.search(fn)
    if not m:
        continue

    date = pd.to_datetime(m.group(1), format='%Y%m%d')
    path = os.path.join(INPUT_DIR, fn)
    paras = [line.strip() for line in open(path, encoding='utf-8') if line.strip()]

    clean = [p for p in paras if p not in seen]
    seen = set(paras)

    score, count = 0.0, 0
    for p in clean:
        text = p.lower()
        if any(kw in text for kw in KEYWORDS):
            score += classify_guidance(p)
        count += 1

    count = max(count, 3)
    rows.append({'date': date, 'guidance_score': score / count})

fomc = pd.DataFrame(rows).sort_values('date')
```

### 4.1) Identificação da data pela regex

* **`pattern = re.compile(r'fomcminutes(\d{8})', flags=re.IGNORECASE)`**

  * Assume que os arquivos têm nomes como `"fomcminutes20200129.txt"` ou `"FOMCMinutes20150805.txt"`.
  * O grupo `(\d{8})` captura um bloco de 8 dígitos (ano/mês/dia), ex.: `20200129`.
  * `ignorecase` significa que tanto `FOMCMinutes...` ou `fomcminutes...` serão capturados.

* **`m = pattern.search(fn)`** verifica se o nome de arquivo bate com esse padrão.

  * Se `m` for `None`, pulamos aquele arquivo (não é uma ata com data no formato esperado).
  * Caso contrário, `m.group(1)` retorna a string de 8 dígitos, e `pd.to_datetime(m.group(1), format='%Y%m%d')` converte em um objeto `Timestamp`.

### 4.2) Leitura e limpeza de parágrafos

* **`paras = [line.strip() for line in open(path, encoding='utf-8') if line.strip()]`**

  * Lê cada linha do arquivo `.txt`, faz `strip()` para retirar espaços em branco e `\n`, e ignora qualquer linha que fique vazia.

* **Remoção de duplicatas**

  * **`clean = [p for p in paras if p not in seen]`**

    * Garante que, se um parágrafo exato já apareceu em uma ata anterior, não vamos duplicar no conjunto atual.
  * **`seen = set(paras)`**

    * Atualiza o conjunto de parágrafos já vistos, para evitar duplicações futuras neste loop.

### 4.3) Cálculo do “guidance\_score” por ata

* Inicializamos:

  * `score = 0.0`
  * `count = 0`

* Para cada parágrafo `p` em `clean`:

  * Transforma para minúsculas: `text = p.lower()`.
  * **Verifica se contém palavra-chave**: `if any(kw in text for kw in KEYWORDS)`.

    * Se `True`, chamamos `classify_guidance(p)`, que retorna um dos valores {+1, 0, –1}.
    * Esse retorno é adicionado ao `score`.
  * `count += 1` → contamos todos os parágrafos (mesmo que não sejam relevantes) para evitar divisão por zero.

* Ao final, `count = max(count, 3)` garante que dividiremos pelo menos por 3 (evita distorção se a ata tiver menos linhas).

* **`guidance_score = score / count`**: média dos scores de todos os parágrafos, ponderada pelo número de parágrafos.

  * É diferente de “média só dos parágrafos relevantes”: aqui penalizamos se uma ata tiver pouquíssimos parágrafos de política.

**Riscos e cuidados**:

* Se uma ata tiver muitos parágrafos não relevantes, o score tenderá a 0 (dividido por um `count` grande), mesmo que existam alguns parágrafos extremamente hawkish/dovish.
* Versão alternativa:

  * Poderíamos usar `count_relevants = número de parágrafos que satisfazem `any(kw\...)`e definir`guidance\_score = score/count\_relevants`(ou`max(count\_relevants, 1)\`), mas isso faria o score “explodir” se houver poucas linhas.

### 4.4) Construção do DataFrame `fomc`

* Ao final do loop, `rows` é uma lista de dicionários: `[{date: ..., guidance_score: ...}, ...]`.
* **`fomc = pd.DataFrame(rows).sort_values('date')`**: gera um DataFrame ordenado por data cronológica.

---

## 5) Suavização e normalização do indicador bruto

```python
# Rolling mean de 3 atas (para suavização leve)
fomc['score_smooth_raw'] = fomc['guidance_score'].rolling(3, center=True, min_periods=1).mean()

# Normalizar em [-1, +1]
fomc['score_smooth'] = MinMaxScaler(feature_range=(-1, 1)).fit_transform(
    fomc[['score_smooth_raw']].fillna(0)
)
```

### 5.1) Por que suavizar?

* Sem suavização, cada ata pode variar excessivamente, sobretudo se uma ata for muito hawkish (score±1) e a próxima for neutra (score≈0). Isso pode gerar muita volatilidade no “Copômetro”.
* O rolling window de 3 (janela de 3 atas consecutivas) faz com que cada ponto seja a média das 3 sessões mais próximas (1 anterior, a atual e 1 posterior), atenuando picos isolados.

### 5.2) Como aplicar

* **`fomc['guidance_score'].rolling(3, center=True, min_periods=1).mean()`**

  * `window=3`: considera 3 valores vizinhos para a média.
  * `center=True`: o valor de “índice i” será a média de i–1, i e i+1.
  * `min_periods=1`: se não existirem 3 vizinhos (início e fim da série), calcula média com pelo menos 1 valor.

* **`MinMaxScaler(feature_range=(-1,1)).fit_transform`**

  * Mapeia linearmente todos os valores de `score_smooth_raw` para o intervalo \[–1, +1].
  * Por exemplo, se `score_smooth_raw` variar entre –0.8 e +0.6, o MinMax normaliza para \[–1, +1].
  * Isso permite que tanto o “Copômetro” quanto as séries de Δ Fed Funds sejam comparáveis no mesmo eixo vertical.

> **Atenção ao overfitting da fase de normalização**
>
> * Aqui usamos toda a série (`fit_transform`) em uma só etapa. Em um cenário de validação (com separação temporal), deveríamos “fit” o scaler apenas nos dados de treino e depois “transform” nos dados de treino *e* teste.
> * Mas como nosso objetivo é rodar apenas “no período completo” (não fizemos divisão temporal estrita aqui), usamos o MinMax em todos os pontos. Em uma versão de teste futuro, vale separar o conjunto e “vazar” esses parâmetros de normalização (leakage) poderia inflar correlações.

---

## 6) Carregar e escalar a série de Δ Fed Funds

```python
df_fed = pd.read_excel(FED_RATE_PATH)
df_fed['date'] = pd.to_datetime(df_fed['date'], errors='coerce')
df_fed = df_fed[(df_fed['date'] >= '2007-10-31') & (df_fed['date'] <= '2025-12-31')]
df_fed['delta_t1'] = df_fed['FEDFUNDS'].shift(-1)
df_fed['fed_scaled'] = MinMaxScaler(feature_range=(-1, 1)).fit_transform(
    df_fed[['delta_t1']].fillna(0)
)
df_fed = df_fed[['date', 'fed_scaled']]
```

### 6.1) Motivação

* **Δ Fed Funds t+1**: nosso alvo de previsão. Ou seja, queremos ver se o “Copômetro” (FOMC-METER) consegue antecipar a variação futura da taxa básica do Fed.
* As atas do FOMC acontecem periodicamente (aprox. 8 vezes por ano). O Fed Funds diário/mensal precisa ser “alinhas” com estas datas.

### 6.2) Passos técnicos

1. **Leitura do Excel**

   * O arquivo `FEDFUNDS.xlsx` deve conter pelo menos colunas:

     * `date` (data da observação)
     * `FEDFUNDS` (valor diário ou mensal da taxa Fed Funds).
2. **Filtragem de período**

   * Mantemos somente datas entre `2007-10-31` e `2025-12-31` (faixa que cobre tanto o pós-crise de 2008 quanto projeções até 2025).
   * Isso é apenas exemplo; você pode ajustar conforme necessidade.
3. **Cálculo de Δ t+1**

   * **`df_fed['delta_t1'] = df_fed['FEDFUNDS'].shift(-1)`**

     * Se a linha i representa a ata do dia `date_i`, então `delta_t1` na linha i será `FEDFUNDS` na linha i+1 (próximo registro).
     * Em termos de séries temporais, isso “desloca” a variável para que o valor real de taxa seja associado ao índice da ata anterior.
4. **Normalização**

   * **`fed_scaled = MinMaxScaler(feature_range=(-1,1)).fit_transform(df_fed[['delta_t1']])`**

     * Cada valor de `delta_t1` (por exemplo, +0.25%, –0.50%, etc.) é projetado no intervalo \[–1, +1], igualando a escala ao “score\_smooth” das atas.

> **Importante**: usar `shift(-1)` sem cuidado pode criar “data leakage” se você futuramente separar em treino/teste. A linha i usa informação da linha i+1, que às vezes pode estar fora do período de interesse no conjunto de teste. Idealmente, para predição “real” você alinharia as datas de ata com a data do Fed ex-post. No entanto, como estamos rodando “no período completo”, esse desalinhamento não foi ressaltado aqui.

---

## 7) Merge das séries por `merge_asof`

```python
base = pd.merge_asof(
    fomc[['date', 'score_smooth']].sort_values('date'),
    df_fed.sort_values('date'),
    on='date', direction='backward'
)
base.dropna(inplace=True)
```

### 7.1) Objetivo do `merge_asof`

* Queremos aliar cada ata do FOMC (em `fomc['date']`) com a observação imediatamente posterior (ou igual) de `fed['fed_scaled']`.

* `pd.merge_asof(..., direction='backward')` faz exatamente isso: para cada data de `fomc`, busca no DataFrame `df_fed` a linha cuja data seja a mais próxima **anterior ou igual**.

  * Exemplo:

    * Ata em `2019-09-18`
    * O `merge_asof` procura, em `df_fed`, a linha de data <= `2019-09-18` mais próxima (por exemplo, 18/09/2019).
    * Se não existir fed em 18/09, busca 17/09 (no caso de diariamente), ou a data mensal mais próxima.

* **Por que “backward”?**

  * Agregamos “ata → fed” com `date_fed <= date_fomc`. Geralmente, queremos prever a mudança da taxa que ocorre depois da ata.
  * Nesse setup, o Δ Fed Funds t+1 já está “shiftado” na linha anterior, então, em geral, quando `base['fed_scaled']` é medido, é a taxa que efetivamente muda após a ata.

### 7.2) `dropna()`

* Remove linhas onde o merge não encontrou correspondência (por exemplo, se a ata for anterior ao primeiro registro do Fed ou posterior ao último).
* Garante que todas as linhas restantes tenham tanto `score_smooth` quanto `fed_scaled`.

---

## 8) Extração de “Surpresa” via filtro HP

```python
cop = base['score_smooth'].values
trend = np.zeros_like(cop)
cycle = np.zeros_like(cop)

for i in range(len(cop)):
    if i < 5:
        trend[i] = cop[i]
        cycle[i] = 0
    else:
        t_i, c_i = hpfilter(cop[:i + 1], lamb=400)
        trend[i] = t_i[-1]
        cycle[i] = cop[i] - t_i[-1]

base['surprise'] = cycle
```

### 8.1) Por que usar HP filter?

* **Objetivo**: separar a série suavizada do Copômetro (`score_smooth`) em duas componentes:

  1. **Tendência (trend)** de longo prazo
  2. **Ciclo (cycle)** de curto prazo, que interpretamos como “surpresa”
* `hpfilter(x, lamb=λ)`

  * Produz `t_i`: o vetor de valores de tendência (mesmo tamanho que `x`).
  * `c_i`: vetor de “ciclo” = `x – trend`.
  * O parâmetro `lamb` regula a suavidade da tendência:

    * Se `λ` for grande (p.ex. 2500 ou 1600), a tendência ficará bem lisa, o ciclo capturará movimentos menores.
    * Se `λ` for pequeno, a tendência ficará mais reativa e o ciclo ficará menor.

### 8.2) Lógica do loop incremental

* O filtro HP tradicional funciona em séries de comprimento fixo. Aqui adaptamos para “uma aplicação progressiva”:

  ```python
  for i in range(len(cop)):
      if i < 5:
          trend[i] = cop[i]; cycle[i] = 0
      else:
          t_i, c_i = hpfilter(cop[:i+1], lamb=400)
          trend[i] = t_i[-1]
          cycle[i] = cop[i] - t_i[-1]
  ```

  * Para as primeiras 5 observações, definimos `trend = cop` e `cycle = 0` (não há histórico suficiente para aplicar HP).
  * A partir do índice `i ≥ 5`, aplicamos HP em `cop[:i+1]` (todas as observações até a i-ésima), extraímos a última posição de `t_i` para `trend[i]` e definimos `cycle[i] = cop[i] – trend[i]`.
  * Esse mecanismo torna o “surprise” dependente só de dados passados (e do próprio `cop[i]`) — respeitando a ideia de predição causal (não usa dados futuros de `cop`).

### 8.3) Riscos de overfitting no HP filter

* O filtro HP não é um modelo de aprendizado, mas pode introduzir ruído se a série for curta.
* O parâmetro `λ=400` foi arbitrário; se você usar `λ` muito baixo, a “surprise” ficará quase sempre zero (porque a tendência acompanhará quase fielmente `cop`). Se for muito alto, o “cycle” ficará muito ruidoso.
* **Sugestão**: testar valores de `λ` (por exemplo, 400, 1000, 1600) e avaliar qual se correlaciona melhor *fora da amostra* com Δ Selic ou Δ Swap. Esse ajuste evita “overfit” acidental do filtro a ruídos de curto prazo.

---

## 9) Construção das features “surprise\_lag1”, “extreme\_shock” e “interaction”

```python
base['surprise_lag1'] = zscore(pd.Series(base['surprise']).shift(1).fillna(0))
thr = np.nanpercentile(np.abs(base['surprise']), 90)
base['extreme_shock'] = (np.abs(base['surprise']) >= thr).astype(int)
base['interaction'] = base['surprise_lag1'] * base['extreme_shock']
```

1. **`surprise_lag1`**

   * `base['surprise'].shift(1)`: desloca a surpresa em 1 ata, ou seja, usamos a surpresa da ata anterior como feature para prever Δ Fed Funds.
   * `fillna(0)`: a primeira linha fica sem valor, então definimos 0 para não quebrar o pipeline.
   * `zscore(...)`: padroniza (média 0 e desvio 1) toda a série.

     * Padronizar é importante para modelos que se beneficiam de features na mesma escala.

2. **`extreme_shock`**

   * Calcula o percentil 90 de `|surprise|`: `thr = np.nanpercentile(np.abs(...), 90)`.
   * Se a `|surprise|` estiver entre os 10% maiores valores históricos, definimos `extreme_shock = 1`, senão 0.
   * Em outras palavras, codificamos “esta ata teve uma surpresa muito inesperada” (parte alta da distribuição).

3. **`interaction`**

   * Simples multiplicação: `interaction = surprise_lag1 * extreme_shock`.
   * Isso significa que, se houve choque extremo (1) na retrospectiva, a magnitude (positiva ou negativa) de `surprise_lag1` será amplificada.
   * Caso contrário, se `extreme_shock = 0`, então toda a linha de interação vira 0.

> **Por que essas features?**
>
> * **`surprise_lag1`**: captura inércia ou persistência da surpresa de política monetária.
> * **`extreme_shock`**: foca apenas em eventos realmente “fora da curva”.
> * **`interaction`**: combinação de magnitude e raridade do choque.

---

## 10) Treinar o RandomForest para prever Δ Fed Funds

```python
X = base[['surprise_lag1', 'extreme_shock', 'interaction']]
y = base['fed_scaled']

rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X, y)
base['fomc_meter'] = rf.predict(X)
```

### 10.1) Motivação

* Queremos um “indicador final” (o FOMC-METER) que traduza as features (`surprise_lag1`, `extreme_shock`, `interaction`) em um valor que reflita a variação futura da taxa (`y = fed_scaled`).
* Neste ponto, usamos **todo o conjunto** para treinar e, imediatamente, predit o próprio conjunto. Isso **não é uma validação real** e satisfaz apenas o propósito de inspecionar se existe correlação forte “in-sample”. Em uma versão robusta, deveríamos fazer:

  * Divisão temporal em treino/teste (por exemplo, treinar até 2019 e testar em 2020–2025).
  * Avaliar métricas *fora da amostra* (correlação, MSE, etc.) para detectar overfitting.

### 10.2) Escolha do RandomForestRegressor

* **200 árvores**: bom ponto de partida para mesclar robustez e tempo de execução.
* Cada árvore vê uma amostra bootstrap (aleatória) de `(X, y)` e um subconjunto aleatório das features (`max_features = sqrt(n_features)` por padrão = √3 ≈ 1–2 features).
* O ensemble de 200 árvores sofre menos com overfitting que uma árvore única muito profunda. Ainda assim, é possível overfit se:

  * O número de features relevantes for muito superior ao número de observações (aqui temos poucos pontos de ata por ano; em 2007–2025 são ≈ 18 anos × 8 atas = \~144 pontos).
  * O modelo memorizar ruído no conjunto.

> **Como diagnosticar overfitting no regressão?**
>
> * **Avaliar erro/função de perda** tanto em treino quanto em teste (via `mean_squared_error` ou correlação *fora da amostra*).
> * Se a performance no treino for muito alta (ex.: correlação > 0.95) e no teste for muito baixa (ex.: < 0.3), é overfitting.
> * É recomendável diminuir `max_depth` ou aumentar `min_samples_leaf`.

---

## 11) Importância de variáveis e residual

```python
imp = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("Importância das features:")
print(imp)

base['residual'] = base['fomc_meter'] - base['fed_scaled']
```

### 11.1) `feature_importances_`

* `rf.feature_importances_` retorna a importância relativa de cada feature (soma de reduções de impureza ponderadas pelo número de amostras que passam por cada nó).
* Em geral, valores mais altos indicam que aquela feature foi mais útil para dividir internamente as árvores e, portanto, tem maior poder explicativo para prever a variável-alvo.

### 11.2) Residual

* **`residual = predicted – real`**

  * Se for perto de zero, o modelo acertou o valor escalado do Fed Funds.
  * Resíduos sistematicamente positivos ou negativos sugerem viés (under/overestimação).
  * Plotamos aqui apenas para visualizar ao longo do tempo se há padrões de erro.

```python
plt.figure(figsize=(10, 4))
plt.plot(base['date'], base['residual'], label='Residual')
plt.axhline(0, color='gray', linestyle='--')
plt.title('Residual FOMC-METER vs ΔFed Funds')
plt.grid(True)
plt.tight_layout()
plt.show()
```

* O gráfico de resíduos ajuda a perceber:

  * Se há “sacadas” isoladas (picos de erro) em determinados períodos (por exemplo, períodos de crise, lockdown, ventos externos etc.).
  * Se existe tendência nos resíduos (por exemplo, sempre subestima no fim do ciclo de alta).

> **Cuidado com overfitting**:
>
> * Se os resíduos ficarem todos muito próximos de zero “in-sample”, mas em validação temporal explodirem, é sinal de que o RF está memorizando peculiaridades de cada ata/observação.

---

## 12) Determinar dinamicamente as zonas Hawkish / Dovish

```python
hawk_th = base['fomc_meter'].quantile(0.75)  # top 25% → hawkish
dove_th = base['fomc_meter'].quantile(0.25)  # bottom 25% → dovish

print(f"Hawkish threshold (75th percentile): {hawk_th:.3f}")
print(f"Dovish threshold (25th percentile): {dove_th:.3f}")
```

* **Racional**:

  * Em vez de fixar valores estáticos (ex.: `> 0.2 → hawkish`, `< −0.2 → dovish`), usamos a própria distribuição de `fomc_meter` para definir regiões:

    * Os 25% maiores valores do indicador (percentil 75%) são considerados “zonas hawkish”.
    * Os 25% menores valores (percentil 25%) se tornam “zonas dovish”.
    * O meio (entre 25% e 75%) é “zona neutra”.

* **Por que isso faz sentido?**

  * Em cada dataset, o range de `fomc_meter` pode se alterar: por exemplo, em um período de “agressiva alta de juros”, quase todo valor do indicador fica alto, então definimos “hawkish” = top 25%.
  * Em outro ciclo de “política altamente dovish ou baixa de juros”, quase tudo está na zona negativa, então “dovish” = bottom 25%.
  * Assim, as zonas se ajustam automaticamente ao contexto histórico e não dependem de um valor fixo que pode ficar ultrapassado.

---

## 13) Plot final: FOMC-METER vs Δ Fed Funds

```python
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(base['date'], base['fomc_meter'], label='FOMC-METER', color='tab:red', linewidth=2)
ax.plot(base['date'], base['fed_scaled'], label='Δ Fed Funds t+1', color='black', linewidth=1.5)

# Zonas definidas pelos percentis
ax.axhspan(hawk_th, base['fomc_meter'].max() + 0.01,
           facecolor='red', alpha=0.1, label='Hawkish Zone')
ax.axhspan(base['fomc_meter'].min() - 0.01, dove_th,
           facecolor='green', alpha=0.1, label='Dovish Zone')

ax.set_title('FOMC-METER vs Δ Fed Funds (Zonas Dinâmicas)', fontsize=14)
ax.set_xlabel('Date')
ax.set_ylabel('Scaled Value')
ax.legend(loc='upper left')
ax.grid(ls=':', alpha=0.5)
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.tight_layout()
plt.show()
```

### 13.1) Descrição dos componentes visuais

1. **Linha vermelha (`fomc_meter`)**:

   * Predições do modelo RandomForest para cada data de ata.
   * É o nosso indicador final que tenta “adivinhar” a direção/tamanho do próximo Δ Fed Funds.

2. **Linha preta (`fed_scaled`)**:

   * A série normalizada de Δ Fed Funds (transformada em \[–1, +1]).
   * Representa o valor “real” que ocorreu após cada ata (date i) até a próxima (date i+1).

3. **Faixas de fundo (zonas)**:

   * **Hawkish Zone (top 25%)**: preenchida em vermelho pálido (`facecolor='red', alpha=0.1`).
   * **Dovish Zone (bottom 25%)**: preenchida em verde pálido (`facecolor='green', alpha=0.1`).
   * O espaço entre `dove_th` e `hawk_th` é a **zona neutra**, mas não pintamos explicitamente (é o fundo branco).

4. **Eixos de tempo**:

   * `mdates.YearLocator(2)` → marcadores a cada 2 anos.
   * `mdates.DateFormatter('%Y')` → formatações apenas do ano (2008, 2010, …).

### 13.2) Interpretação

* Se o **FOMC-METER** (vermelho) estiver acima do percentil 75% (área vermelha no topo), interpretamos que o modelo está “muito hawkish” e provavelmente a próxima Δ Fed Funds será alta.
* Se estiver abaixo do percentil 25% (área verde no fundo), interpretamos como “muito dovish” e prevemos redução/tendência de queda de juros.
* Entre esses percentis, o modelo está neutro.
* A proximidade (ou divergência) entre as duas curvas (vermelha vs preta) indica o quão bem o modelo consegue seguir a série real.

> **Observação sobre overfitting**:
>
> * Se o FOMC-METER ajustar “perfeitamente” a linha preta *in-sample* (correlação ≈ 0.9+), mas for muito errático ou não generalizar nos períodos de teste, há evidência de overfitting.
> * As zonas dinâmicas ajudam a avaliar se o modelo apenas captura a distribuição estatística ou se realmente capta sinais de texto. Se em muitos pontos “paintar” hawkish mas a série real não subir, possivelmente o classificador de guidance está “culando” (rotulando tudo como hawkish), ou o RF está se apoiando demais em ruído nos dados.

---

## 14) Análise de correlação

```python
corr_meter = pearsonr(base['fomc_meter'], base['fed_scaled'])[0]
print(f"Correlation FOMC-METER vs Δ Fed Funds: {corr_meter:.3f}")

for feat in ['surprise_lag1', 'extreme_shock', 'interaction']:
    c = pearsonr(base['fed_scaled'], base[feat])[0]
    print(f"Correlation ΔFed Funds vs {feat}: {c:.3f}")
```

1. **`pearsonr(x, y)`** retorna `(correlation_coefficient, p_value)`.

2. A correlação principal é entre **`fomc_meter`** (previsão) e **`fed_scaled`** (valor real).

   * Um coeficiente perto de 1 indica forte associação linear (in-sample).
   * Mas cuidado: correlação alta in-sample não garante poder preditivo out-of-sample.

3. Em seguida, checamos correlação de **cada feature isolada** (`surprise_lag1`, `extreme_shock`, `interaction`) contra a série real `fed_scaled`.

   * Se alguma feature isolada tiver correlação alta, talvez signifique que nosso RF pode estar basicamente usando só uma variável dominante.
   * Por exemplo, se `surprise_lag1` tiver correlação 0.8 com `fed_scaled`, talvez nem valha a pena usar as outras 2 features.

4. **Cuidado com correlação espúria**:

   * Corrigir outliers antes de calcular correlação, pois pontos extremos podem distorcer muito o coeficiente.
   * Avaliar `scatter plots` para ver se a relação é de fato linear ou se existem clusters (ex.: um cluster em crise, outro em normalidade).

---

## 15) Exportação final

```python
base.to_excel(os.path.join(OUTPUT_DIR, 'fomc_meter_output_adjusted.xlsx'), index=False)
print("Export completed.")
```

* **`base`** contém colunas:

  * `date` (data da ata)
  * `score_smooth` (Copômetro suavizado e normalizado)
  * `fed_scaled` (Δ Fed Funds normalizado)
  * `surprise`, `surprise_lag1`, `extreme_shock`, `interaction` (todas as features)
  * `fomc_meter` (previsões do RF)
  * `residual` (erro in-sample)
* **Exportar para Excel** permite compartilhar os dados e gráficos com colegas ou carregar em PowerPoint, GitHub, etc.

---

## 16) Considerações gerais sobre Overfitting e Robustez

1. **Separação treino/teste automática**

   * No classificador de guidance, usamos `train_test_split` para garantir que a avaliação de acurácia não aconteça “in-sample”.
   * Na parte do RF que prevê Δ Fed Funds, usamos TODO o conjunto para treinar e predizer (in-sample). Em um cenário ideal, deveríamos:

     * Dividir o período (ex.: até 2018 como treino, 2019–atual como teste).
     * Ajustar hiperparâmetros do RF (por exemplo, profundidade máxima, número de estimadores, amostra mínima por folha, etc.) usando validação cruzada somente no grupo de treino.
     * Avaliar performance só no conjunto de teste (f2-score, MSE, R², correlação).

2. **Complexidade do modelo**

   * O RF, por padrão, tentará reduzir o erro de treino ao máximo. Se tivermos poucas observações (<200 atas) e 3–4 features, 200 árvores já podem “memorizar” ruídos e gerar overfitting.
   * Para reduzir o overfitting no RF, podemos:

     * **Diminuir `max_depth`** (por exemplo, `max_depth=5` ou `10`).
     * **Aumentar `min_samples_leaf`** (por exemplo, `10` ou `20`). Isso força cada folha a ter pelo menos X observações.
     * **Reduzir `n_estimators`** se notarmos overfitting extremo (embora às vezes mais árvores ajudem a estabilizar).

3. **Features adicionais e engenharia de variáveis**

   * Dependendo do contexto, talvez valha a pena incluir outras features de texto:

     * **Sentiment geral do parágrafo** (usando VADER ou BERT para sentimento).
     * **Número de menções específicas** (ex.: contar quantas vezes “inflation” aparece).
     * **Comprimento do parágrafo** (muitas vezes parágrafos longos contêm mais informações sobre riscos, o que pode ter peso).
   * Cada nova feature deve ser avaliada quanto à **colinearidade** e à real “sinalização” para a série de taxas.

4. **Suavização vs HP filter**

   * O filtro HP pode superajustar quando a série é curta. Se trocarmos por suavização rolling, corremos o risco de torná-lo “muito lento” para capturar mudanças súbitas.
   * A estratégia que adotamos (rolling de 3 atas + HP com λ=400 nos últimos pontos) é um **compromisso intermediário**:

     * O rolling de 3 já ameniza picos isolados,
     * O HP pega a diferença local entre a sessão atual e a tendência de curto prazo que se formou nas 5–10 sessões anteriores.

5. **Validação temporal (Rolling Windows)**

   * Como alternativa ao “usar tudo para treinar e predizer”, você pode simular:

     ```python
     for split_date in ['2016-12-31', '2018-12-31', '2020-12-31']:
         train_data = base[base['date'] <= split_date]
         test_data  = base[base['date'] >  split_date]
         X_tr = train_data[['surprise_lag1','extreme_shock','interaction']]
         y_tr = train_data['fed_scaled']
         X_te = test_data[['surprise_lag1','extreme_shock','interaction']]
         y_te = test_data['fed_scaled']
         rf_t = RandomForestRegressor(...).fit(X_tr,y_tr)
         y_pred = rf_t.predict(X_te)
         corr = pearsonr(y_pred, y_te)[0]
         print(f"Teste pós {split_date}: correlação = {corr:.3f}")
     ```
   * Isso dá uma **métrica robusta fora da amostra**, mostrando se o modelo se comporta bem em períodos não vistos.

6. **Detecção de viés no classificador de guidance**

   * Se a maioria dos parágrafos do FOMC estiver sendo rotulada como “neutral” ou “dovish”, o `score_smooth` ficará sempre próximo de zero ou < 0.
   * Vale extrair algumas amostras manualmente:

     ```python
     for p in sample_paras:
         print(p, "→", classify_guidance(p))
     ```
   * Se notar vieses (por exemplo, classifica tudo como dovish), é sinal de que o classificador precisa de mais exemplos “hawkish” ou “neutral” de atas do Fed (em vez de frases do Copom em português).

7. **Análise de indicadores de desempenho**

   * Além da **correlação**, avalie:

     * **Mean Squared Error (MSE)** ou **RMSE**: para medir tamanho bruto do erro.
     * **R²**: quão bem a variação do Fed Funds é explicada pelo modelo.
     * **Plot de scatter** entre `fomc_meter` e `fed_scaled` para ver se há outliers extremados ou relação não linear.
   * Se perceber que `surprise_lag1` captura boa parte do sinal (feature importance ≈ 0.9), talvez o HP filter esteja trazendo o grosso da informação – e o RF pode estar memorizando relacionamento linear entre `surprise_lag1` e `fed_scaled`.

---

### Resumo das etapas e cuidados para evitar overfitting

1. **Treinar Classificador de Guidance (RFClassifier)**

   * **Dividir em treino/teste (20% teste)** para medir acurácia real.
   * Ajustar hiperparâmetros se houver gap grande entre treino/teste (ex.: limitar máxima profundidade das árvores).

2. **Filtrar parágrafos das atas**

   * Usar um conjunto bem abrangente de `KEYWORDS`, para que o score tenha variação real.
   * Evitar regras que façam `score` quase sempre zero (ex.: se “monetary” não aparecer em certas atas, o score será 0).

3. **Construir “guidance\_score” médio**

   * Dividir apenas pelos parágrafos filtrados ou por todos os parágrafos?
   * O script divide por todos para penalizar atas com poucas menções, mas também pode “diluir” sinal.

4. **Suavizar (rolling)**

   * Evitar suavizar demais, pois pode achatar sinais importantes.
   * Ajustar janela (2, 3 ou 4) e confirmar empiricamente qual gera melhor correlação *fora da amostra*.

5. **Calcular “Surpresa” – HP filter**

   * Escolher λ adequado, testando valores (ex.: 400, 1000, 1600, 2500) com validação temporal.
   * Esse ciclo deve conter variações realmente residuais, não “filtro de ruído”.

6. **Gerar features e Treinar RFRegressor**

   * **Separar treino/teste** por data (não usar tudo in-sample).
   * Ajustar:

     * `n_estimators`,
     * `max_depth`,
     * `min_samples_leaf`,
     * possivelmente `max_features='sqrt'` ou outro método.

7. **Avaliar performance fora da amostra**

   * Calcular correlação, MSE e R² em janelas distintas (por exemplo, treinando até 2018 e testando 2019–2025).
   * Verificar se as importâncias de features se mantêm estáveis ou mudam radicalmente em diferentes janelas (sinal de instabilidade).

8. **Plot e interpretações**

   * O “poder de predição” do FOMC-METER será perceptível se a linha vermelha ficar próxima da preta nos períodos de teste (fora da amostra).
   * Se houver grandes discrepâncias, revisar:

     * Base de frases de guidance (talvez precisamos de exemplos *específicos* do FOMC, não apenas frases genéricas de Copom),
     * Pipeline de filtros e suavizações.

---

### Considerações finais sobre robustez

* **Garanta exemplos balanceados** no `df_guid`. Se tiver muito mais frases “dovish” que “hawkish”, o classificador tenderá a enviesar tudo para “dovish”.
* **Aumentar a base de frases rotuladas** (de 200 para 500 ou 1000) pode ajudar a capturar melhor nuances de linguagem em reuniões do Fed.
* **Experimentar outro modelo de embedding**, por exemplo “`distiluse-base-multilingual-cased-v2`” ou “`xlm-roberta-base`”, para verificar se as embeddings melhoram a classificação de frases em inglês.
* **Teste de sensibilidade**: altere `KEYWORDS` e observe como o `raw_score` muda. Às vezes, incluir palavras como “forex”, “quantitative easing”, “liquidity” etc. ajuda a capturar declarações de política monetária que não contenham explicitamente “interest” ou “inflation”.
* **Não confunda correlação in-sample com real poder preditivo**. Sempre faça um teste temporal de previsão para evitar ver “fantasmas” no seu modelo.

---

Ao seguir cuidadosamente cada uma dessas etapas – separando treino e teste, ajustando hiperparâmetros, testando λ do HP filter, balanceando amostras – você reduz bastante a chance de overfitting e garante que o FOMC-METER capture de fato o tom das atas, em vez de memorizar padrões específicos de um período. Se tiver dúvidas em algum ponto específico (por exemplo, escolha de λ do HP ou refinamento de KEYWORDS), posso detalhar ainda mais.




'''