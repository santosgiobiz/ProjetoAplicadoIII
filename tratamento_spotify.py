"""
Tratamento do Spotify Tracks Dataset (Projeto Aplicado III, Etapa 2, seção 4.3).

Uso:
    1. Baixe o dataset.csv do Kaggle (maharshipandya/-spotify-tracks-dataset)
    2. Coloque na mesma pasta deste script (ou ajuste CAMINHO_ENTRADA)
    3. python tratamento_spotify.py

Saídas:
    spotify_limpo.csv     -> base limpa, escala original (para consulta/EDA)
    spotify_tratado.csv   -> base limpa + features escaladas em [0, 1] (para o k-NN)
"""
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

CAMINHO_ENTRADA = "dataset.csv"
PERCENTIL_INF, PERCENTIL_SUP = 0.01, 0.99  # ajustável: define o corte de outliers

FEATURES = [
    "danceability", "energy", "loudness", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo", "popularity",
]

# --- Leitura ---------------------------------------------------------------
df = pd.read_csv(CAMINHO_ENTRADA)
df = df.drop(columns=["Unnamed: 0"], errors="ignore")  # índice que vem no CSV do Kaggle
print(f"Linhas originais: {len(df):,}")

# --- 1. Duplicatas: track_name + artists, mantendo a maior popularity ------
df = (
    df.sort_values("popularity", ascending=False)
      .drop_duplicates(subset=["track_name", "artists"], keep="first")
)
print(f"Após remover duplicatas: {len(df):,}")

# --- 2. Nulos (< 0,1%): remoção dos registros incompletos ------------------
df = df.dropna()
print(f"Após remover nulos: {len(df):,}")

# --- 3. Outliers em loudness e duration_ms: corte por percentis ------------
for col in ["loudness", "duration_ms"]:
    baixo, alto = df[col].quantile([PERCENTIL_INF, PERCENTIL_SUP])
    df = df[df[col].between(baixo, alto)]
print(f"Após cortar outliers: {len(df):,}")

df = df.reset_index(drop=True)
df.to_csv("spotify_limpo.csv", index=False)

# --- 4. Escalonamento (MinMax) do vetor de características ------------------
scaler = MinMaxScaler()
df_escalado = df.copy()
df_escalado[FEATURES] = scaler.fit_transform(df[FEATURES])
df_escalado.to_csv("spotify_tratado.csv", index=False)

print("\nArquivos gerados: spotify_limpo.csv e spotify_tratado.csv")
print(df_escalado[FEATURES].describe().loc[["min", "max"]])
