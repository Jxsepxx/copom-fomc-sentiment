# -*- coding: utf-8 -*-
import os, re
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sentence_transformers import SentenceTransformer
from scipy.stats import zscore, pearsonr
from statsmodels.tsa.filters.hp_filter import hpfilter
import matplotlib.dates as mdates
import seaborn as sns

nltk.download('punkt')

BASE_DIR      = Path(__file__).parent
INPUT_DIR     = BASE_DIR / 'atas_copom' / 'txt_copom'
SEL_RATE_PATH = BASE_DIR / 'Base_SELIC.xlsx'
SWAP2_PATH    = BASE_DIR / 'Base_2Y.xlsx'
GUIDANCE_PATH = BASE_DIR / 'copom_additional_refined_902.xlsx'
OUTPUT_DIR    = BASE_DIR / 'outputs'
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------
# Bloco 1: Treinamento do classificador de guidance 
# ---------------------------------------------------------
df_guid = pd.read_excel(GUIDANCE_PATH)
df_guid = df_guid.dropna(subset=['texto', 'tipo'])
df_guid['tipo'] = df_guid['tipo'].str.lower().str.strip()
df_guid = df_guid[df_guid['tipo'].isin(['hawkish', 'dovish', 'neutral'])]


model_st = SentenceTransformer('paraphrase-MiniLM-L6-v2')
embeddings_guid = model_st.encode(df_guid['texto'].tolist(), convert_to_tensor=False)


Xg_train, Xg_test, yg_train, yg_test = train_test_split(
    embeddings_guid, df_guid['tipo'], test_size=0.2, random_state=42
)
clf_guid = RandomForestClassifier(
    bootstrap=True, criterion='gini', n_estimators=200, oob_score=True, random_state=42
) 
clf_guid.fit(Xg_train, yg_train)

print(f"OOB Score do classificador de guidance: {clf_guid.oob_score_:.3f}")  
SCORE_MAP = {'hawkish': 1, 'neutral': 0, 'dovish': -1}
def classify_guidance_batch(text_list):
    """
    Classifica uma lista de textos em lote para máxima eficiência.
    Retorna uma lista de scores numéricos.
    """
    if not text_list:
        return []
    
    # 1. Codifica todos os textos de uma vez
    embeddings = model_st.encode(text_list, convert_to_tensor=False)
    
    # 2. Faz todas as predições de uma vez
    predictions = clf_guid.predict(embeddings)
    
    # 3. Mapeia os resultados para os scores
    scores = [SCORE_MAP[pred] for pred in predictions]
    
    return scores

# ---------------------------------------------------------
# Bloco 2: Processamento das atas
# ---------------------------------------------------------
files = sorted(
    [f for f in os.listdir(INPUT_DIR) if f.lower().endswith('.txt')],
    key=lambda fn: pd.to_datetime(re.search(r'(\d{2})(\d{2})(\d{4})', fn).group(0), format='%d%m%Y')
)

rows, prev = [], set()
keywords = ['expectations', 'guidance', 'forward guidance', 'forward guidance on the path of interest rates','monetary','monetary policy'
            ,'monetary policy strategy', 'monetary policy stance', 'policy rate', 'policy interest rate', 'interest rate path'
            ]
for fn in files:
    date = pd.to_datetime(re.search(r'(\d{2})(\d{2})(\d{4})', fn).group(0), format='%d%m%Y')
    with open(os.path.join(INPUT_DIR, fn), encoding='utf-8') as f:
        paras = [line.strip() for line in f if line.strip()]
    clean = [p for p in paras if p not in prev]
    prev = set(paras)
    relevant_paras = [p for p in clean if any(kw in p.lower() for kw in keywords)]
    # 2. Classifica todos de uma vez com a nova função
    scores = classify_guidance_batch(relevant_paras)

    # 3. Calcula a média dos scores (de forma mais clara)
    if scores:
        final_score = sum(scores) / len(scores)
    else:
        final_score = 0  # Score neutro se nenhum parágrafo relevante for encontrado
    rows.append({'date': date, 'guidance_dir': final_score})

df = pd.DataFrame(rows).sort_values('date')
df['cop_smoothed'] = df['guidance_dir'].rolling(2, center=True, min_periods=1).mean()
df['cop_smoothed'] = MinMaxScaler(feature_range=(-1,1)).fit_transform(df[['cop_smoothed']].fillna(0))

# ---------------------------------------------------------
# Bloco 3: Construção das features econômicas (SEL e SWAP)
# ---------------------------------------------------------
sel = pd.read_excel(SEL_RATE_PATH)
sel['date'] = pd.to_datetime(sel['data'])
sel['delta_t1'] = sel['delta_selic_t1 verdadeiro'].shift(-1)
sel['selic_scaled'] = MinMaxScaler(feature_range=(-1,1)).fit_transform(sel[['delta_t1']].fillna(0))
sel = sel[['date','selic_scaled']]

swap2 = pd.read_excel(SWAP2_PATH)
swap2['date'] = pd.to_datetime(swap2['date'])
swap2['swap2y_scaled'] = MinMaxScaler(feature_range=(-1,1)).fit_transform(swap2[['swap2y']].fillna(0))
swap2 = swap2[['date','swap2y_scaled']]

# ---------------------------------------------------------
# Bloco 4: Merge e detecção inicial de possíveis data leakage
# ---------------------------------------------------------


base = pd.merge_asof(df.sort_values('date'), sel.sort_values('date'), on='date')
base = pd.merge_asof(base.sort_values('date'), swap2.sort_values('date'), on='date')
base.dropna(inplace=True)

print("Intervalo de datas em 'df' (guias):", df['date'].min(), "-", df['date'].max())  
print("Intervalo de datas em 'sel':", sel['date'].min(), "-", sel['date'].max())       
print("Intervalo de datas em 'swap2':", swap2['date'].min(), "-", swap2['date'].max()) 
print("Intervalo de datas em 'base' após merge:", base['date'].min(), "-", base['date'].max())  


# ---------------------------------------------------------
# Bloco 5: Cálculo da “surpresa” com HP filter (sem alterações)
# ---------------------------------------------------------
cop = base['cop_smoothed'].values
trend = np.zeros_like(cop)
cycle = np.zeros_like(cop)
for i in range(len(base)):
    if i < 5:
        trend[i] = cop[i]
        cycle[i] = 0
    else:
        t_i, c_i = hpfilter(cop[:i+1], lamb=2500)
        trend[i] = t_i[-1]
        cycle[i] = cop[i] - trend[i]
base['surprise_hp'] = cycle
base['surprise_lag1'] = zscore(pd.Series(base['surprise_hp']).shift(1).fillna(0))
thr = np.nanpercentile(np.abs(base['surprise_hp']), 90)
base['extreme_shock'] = (np.abs(base['surprise_hp']) >= thr).astype(int)
base['interaction'] = base['surprise_lag1'] * base['extreme_shock']

# ---------------------------------------------------------
# Bloco 6: Preparação do conjunto para o modelo e inspeção de correlação
# ---------------------------------------------------------
df_plot = base[['date','swap2y_scaled','selic_scaled','surprise_lag1','extreme_shock','interaction']].dropna().reset_index(drop=True)
corr_surprise_swap2 = pearsonr(df_plot['surprise_lag1'], df_plot['swap2y_scaled'])[0]
corr_extreme_swap2 = pearsonr(df_plot['extreme_shock'], df_plot['swap2y_scaled'])[0]
corr_inter_swap2 = pearsonr(df_plot['interaction'], df_plot['swap2y_scaled'])[0]
print(f"Correlação entre surprise_lag1 e Swap2y_scaled: {corr_surprise_swap2:.3f}")  
print(f"Correlação entre extreme_shock e Swap2y_scaled: {corr_extreme_swap2:.3f}")  
print(f"Correlação entre interaction e Swap2y_scaled: {corr_inter_swap2:.3f}")       

X = df_plot[['surprise_lag1','extreme_shock','interaction']]
y = df_plot['selic_scaled']

# ---------------------------------------------------------
# Bloco 7: Divisão Treino/Teste e avaliação de overfitting
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, shuffle=False  
)

# Treinamento do modelo com OOB e observação de pontuações
rf_model = RandomForestRegressor(n_estimators=200, oob_score=True, random_state=42)
rf_model.fit(X_train, y_train)

# Métricas de treino e teste (R²)
r2_train = rf_model.score(X_train, y_train)
r2_test = rf_model.score(X_test, y_test)
print(f"R² no conjunto de treino: {r2_train:.3f}")   
print(f"R² no conjunto de teste: {r2_test:.3f}")     
print(f"OOB R² aproximado (estimativa interna): {rf_model.oob_score_:.3f}")  

cv_scores = cross_val_score(
    RandomForestRegressor(n_estimators=200, random_state=42),
    X_train, y_train,
    cv=5, scoring='r2'
)
print(f"Cross‐validation R² media (5 folds): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")  

# ---------------------------------------------------------
# Bloco 8: Gerar curvas de aprendizado (learning curves) para diagnosticar overfitting
# ---------------------------------------------------------
train_sizes, train_scores, valid_scores = learning_curve(
    RandomForestRegressor(n_estimators=200, random_state=42),
    X_train, y_train,
    train_sizes=np.linspace(0.1, 1.0, 5),
    cv=3, scoring='r2', shuffle=False
)

train_scores_mean = np.mean(train_scores, axis=1)
valid_scores_mean = np.mean(valid_scores, axis=1)

plt.figure(figsize=(8, 5))
plt.plot(train_sizes, train_scores_mean, label='R² Treino')
plt.plot(train_sizes, valid_scores_mean, label='R² Validação', linestyle='--')
plt.xlabel('Tamanho do Treino')
plt.ylabel('R²')
plt.title('Learning Curve - RandomForestRegressor')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'learning_curve_rf.png'))
plt.show()

# ---------------------------------------------------------
# Bloco 9: Previsão no conjunto completo e análise de importância
# ---------------------------------------------------------
X_full = X
y_full = y

rf_full = RandomForestRegressor(n_estimators=200, random_state=42)
rf_full.fit(X_full, y_full)
df_plot['copometro_opt'] = rf_full.predict(X_full)

imp  = pd.DataFrame({
    'feature': X_full.columns,
    'importance': rf_full.feature_importances_
}).sort_values('importance', ascending=False)
print("Importâncias das features (modelo final):")
print(imp)
df_plot.to_excel("final.xlsx")
df_plot['residual'] = df_plot['copometro_opt'] - df_plot['swap2y_scaled']
plt.figure(figsize=(10,4))
plt.plot(df_plot['date'], df_plot['residual'])
plt.axhline(0, color='gray', linestyle='--')
plt.title('Erro Residual do Modelo (Copômetro - Swap2Y)')
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'residual_plot_full.png'))
plt.show()

plt.figure(figsize=(8, 5))
plt.xticks(range(len(imp)), imp['feature'], rotation=45)
plt.bar(range(len(imp)), imp['importance'], color='tab:blue')
plt.title('Importância das Features no Modelo Copômetro')
plt.xlabel('Features')
plt.ylabel('Importância')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'importancia_features_copometro_rf.png'))
plt.savefig('2.png')
plt.show()

# ---------------------------------------------------------
# Bloco 10: Correlações finais
# ---------------------------------------------------------
corr_swap2y = pearsonr(df_plot['copometro_opt'], df_plot['swap2y_scaled'])[0]
corr_selic = pearsonr(df_plot['copometro_opt'], df_plot['selic_scaled'])[0]
print(f"Correlação geral com Swap2y: {corr_swap2y:.3f}")
print(f"Correlação geral com Selic: {corr_selic:.3f}")

# ---------------------------------------------------------
# Bloco 11: Plot comparativo completo
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(12,4))
ax.plot(df_plot.date, df_plot.copometro_opt, label='Copômetro', color='tab:red')
ax.plot(df_plot.date, df_plot.swap2y_scaled, label='Δ Swap2y t+1', color='black')
ax.plot(df_plot.date, df_plot.selic_scaled, label='Δ Selic t+1', color='blue', linestyle='--')
ax.set_title(f'Copômetro vs Swap2y e Selic (Completo)')
ax.axhspan(0.2, 1.0, facecolor='red', alpha=0.1)
ax.axhspan(-0.2, 0.2, facecolor='gray', alpha=0.1)
ax.axhspan(-1.0, -0.2, facecolor='green', alpha=0.1)
ax.set_xlabel('Data'); ax.set_ylabel('Valor (scaled)')
ax.legend(loc='upper left'); ax.grid(ls=':', alpha=0.5)
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'copometro_rf_fullswap_selic.png'))
plt.savefig('4.png')
plt.show()

fig, ax = plt.subplots(figsize=(12,4))
ax.plot(df_plot.date, df_plot.copometro_opt, label='Copômetro', color='tab:red')
ax.plot(df_plot.date, df_plot.selic_scaled, label='Δ Selic t+1', color='black')
ax.set_title(f'Copômetro vs Δ Selic (Completo)')
ax.axhspan(0.2, 1.0, facecolor='red', alpha=0.1)
ax.axhspan(-0.2, 0.2, facecolor='gray', alpha=0.1)
ax.axhspan(-1.0, -0.2, facecolor='green', alpha=0.1)
ax.set_xlabel('Data'); ax.set_ylabel('Valor (scaled)')
ax.legend(loc='upper left'); ax.grid(ls=':', alpha=0.5)
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'copometro_rf_fullselic.png'))
plt.savefig('3.png')
plt.show()

# ---------------------------------------------------------
# Bloco 12: Rolling Windows (sem alteração, mas adicionei salvamento de métricas)
# ---------------------------------------------------------
print("\nRolling window validations:")
start_years = [2011, 2013, 2015, 2017, 2019]
rolling_results = []
for start in start_years:
    train_end = pd.to_datetime(f"{start}-12-31")
    test_start = pd.to_datetime(f"{start+1}-01-01")
    train = df_plot[df_plot['date'] <= train_end]
    test  = df_plot[df_plot['date'] >= test_start]

    if len(test) < 10: continue

    X_train = train[['surprise_lag1','extreme_shock','interaction']]
    y_train = train['selic_scaled']
    X_test  = test[['surprise_lag1','extreme_shock','interaction']]
    y_test  = test['selic_scaled']

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    corr = pearsonr(y_pred, y_test)[0]
    mse = np.mean((y_pred - y_test)**2)
    rolling_results.append({'start': start+1, 'corr': corr, 'mse': mse})
    print(f"Treinando até {train_end.date()}, testando em {test_start.date()}+: correlação = {corr:.3f}, MSE = {mse:.5f}")  

    plt.figure(figsize=(10,4))
    plt.plot(test['date'], y_test, label='Swap2y Real', color='black')
    plt.plot(test['date'], y_pred, label='Copômetro Previsto', color='red')
    plt.title(f'Previsão RF: Janela {start+1}+')
    plt.axhspan(0.2, 1.0, facecolor='red', alpha=0.1)
    plt.axhspan(-0.2, 0.2, facecolor='gray', alpha=0.1)
    plt.axhspan(-1.0, -0.2, facecolor='green', alpha=0.1)
    plt.xlabel('Data')
    plt.ylabel('Valor (scaled)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'rolling_window_{start+1}.png'))
    plt.show()
# Supondo que você já tenha df_plot completo com copometro_opt, swap2y_scaled, surprise_lag1

# 1) Criar sinal contínuo → categórico
def classificar_direcao(valor, tol=0.0):
    if valor > tol:
        return 'hawkish'
    elif valor < -tol:
        return 'dovish'
    else:
        return 'neutral'

tol = 0.0
df_plot['sinal_direcao'] = df_plot['copometro_opt'].apply(lambda x: classificar_direcao(x, tol))

# 2) Gerar direção real de swap2y (para backtest histórico)
df_plot['swap2y_next'] = df_plot['swap2y_scaled'].shift(-1)
df_plot['direcao_real_swap2y'] = df_plot.apply(
    lambda row: 'hawkish' if row['swap2y_next'] > row['swap2y_scaled']
                else ('dovish' if row['swap2y_next'] < row['swap2y_scaled']
                      else 'neutral'),
    axis=1
)

# 3) Avaliar acurácia direcional (sem neutros)
comparacao = df_plot.dropna(subset=['sinal_direcao', 'direcao_real_swap2y'])
mask = (comparacao['sinal_direcao'] != 'neutral') & (comparacao['direcao_real_swap2y'] != 'neutral')
acuracia = (comparacao.loc[mask, 'sinal_direcao'] == comparacao.loc[mask, 'direcao_real_swap2y']).mean()
print(f"Acurácia direcional (sem neutros): {acuracia:.2%}")

# 4) Visualizar indicador
import matplotlib.pyplot as plt

plt.figure(figsize=(12,4))
plt.plot(df_plot.date, df_plot['copometro_opt'], label='Copômetro (contínuo)', color='tab:red')
plt.axhspan(0, df_plot['copometro_opt'].max(), facecolor='red', alpha=0.1)
plt.axhspan(df_plot['copometro_opt'].min(), 0, facecolor='green', alpha=0.1)
plt.axhline(0, color='black', linestyle='--')
plt.title('Copômetro – Indicador de Direção (contínuo)')
plt.xlabel('Data'); plt.ylabel('Valor (scaled)')
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig('1.png')
plt.show()

# 5) Gráfico categórico
mapping = {'hawkish': 1, 'neutral': 0, 'dovish': -1}
df_plot['sinal_num'] = df_plot['sinal_direcao'].map(mapping)

plt.figure(figsize=(12,3))
plt.step(df_plot.date, df_plot['sinal_num'], where='mid', color='tab:blue')
plt.yticks([-1, 0, 1], ['dovish', 'neutral', 'hawkish'])
plt.title('Sinal de Direção – Copômetro (categoria)')
plt.xlabel('Data')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('2.png')
plt.show()

rolling_df = pd.DataFrame(rolling_results)
rolling_df.to_excel(os.path.join(OUTPUT_DIR, 'rolling_window_metrics.xlsx'), index=False)  

print("Exportação final concluída.")
