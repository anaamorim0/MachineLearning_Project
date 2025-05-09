import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from knn import KNN
from noise import noise
from outliers import outliers
from metricas import metricas
from matriz_conf import matriz_confusao
from grafico_metricas import grafico_metricas


# Resolver o problema de células vazias na database:
    # se a linha for toda vazia, apaga a linha
    # senão completa com a média da coluna
def celulas_vazias(X):
    # remover colunas todas a NaN
    X = X[:, ~np.isnan(X).all(axis=0)]

    # se ainda existirem celulas vazias:
    if np.isnan(X).any():
        col_means = np.nanmean(X, axis=0)   # faz a média
        inds = np.where(np.isnan(X))    # encontra os indices onde existem NaN
        X[inds] = np.take(col_means, inds[1])   # substitui pelo valor médio
    
    return X


#------------------------------------------------------------------------------------------

dataset = input("Insira o path do dataset: ")
nivel_noise = float(input("Insira o nível de noise (entre 0 e 1): "))
nivel_outliers = float(input("Insira o nível de outliers (entre 0 e 1): "))

df = pd.read_csv(dataset)

y = df.iloc[:, -1].values   # y contem os valores da última coluna = classe
X = pd.get_dummies(df.iloc[:, :-1]).values.astype(float)    # X fica com todas as colunas menos a última, usa get_dummies() para fazer One-Hot Encoding

X = celulas_vazias(X)

indices = np.arange(len(X))
np.random.seed(42)  # faz com que o embaralhamento seja sempre o mesmo quando se corre o código
np.random.shuffle(indices)  # embaralha a ordem dos indices
# reorganiza os dados
X = X[indices]
y = y[indices]

# dividir o dataset em 80% treino e 20% teste
split_idx = int(0.8 * len(X))
X_train = X[:split_idx]
y_train = y[:split_idx]
X_test = X[split_idx:]
y_test = y[split_idx:]

#-----------------------------------------------------
# Dataset sem alterações
modelo = KNN(k=5)
modelo.fit(X_train, y_train)
prediction = modelo.predict(X_test)
accuracy, precision, recall, f1_score = metricas(y_test, prediction)

# Datset com noise
y_noise = noise(y_train, nivel_noise)
modelo_noise = KNN(k=5)
modelo_noise.fit(X_train, y_noise)
prediction_noise = modelo_noise.predict(X_test)
accuracy_noise, precision_noise, recall_noise, f1_noise = metricas(y_test, prediction_noise)

# Dataset com outliers
X_outliers, y_outliers = outliers(X_train, y_train, nivel_outliers, magnitude=10)
modelo_outliers = KNN(k=5)
modelo_outliers.fit(X_outliers, y_outliers)
prediction_outliers = modelo_outliers.predict(X_test)
accuracy_outliers, precision_outliers, recall_outliers, f1_outliers = metricas(y_test, prediction_outliers)

#-----------------------------------------------------------------------------------
# Dataset sem alterações
print("\nDATASET SEM ALTERAÇÕES")
print("Previsões:", [str(p) for p in prediction])
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1_score:.4f}")
fig, ax = plt.subplots(figsize=(5, 2.5))
matriz_confusao(y_test, prediction, ax, title="Matriz de Confusão - Dataset sem Alterações")
plt.tight_layout()
plt.show()

# Dataset com adição de Noise
print("\nDATASET COM NOISE")
print("Previsões:", [str(p) for p in prediction_noise])
print(f"Accuracy com {int(nivel_noise * 100)}% de ruído: {accuracy_noise:.4f}")
print(f"Precision: {precision_noise:.4f}")
print(f"Recall: {recall_noise:.4f}")
print(f"F1-score: {f1_noise:.4f}")
fig, ax = plt.subplots(figsize=(5, 2.5))
matriz_confusao(y_test, prediction_noise, ax, title="Matriz de Confusão - Dataset com Noise")
plt.tight_layout()
plt.show()

# Dataset com adição de Outliers
print("\nDATASET COM OUTLIERS")
print("Previsões:", [str(p) for p in prediction_outliers])
print(f"Accuracy com {int(nivel_outliers * 100)}% de outliers: {accuracy_outliers:.4f}")
print(f"Precision: {precision_outliers:.4f}")
print(f"Recall: {recall_outliers:.4f}")
print(f"F1-score: {f1_outliers:.4f}")
fig, ax = plt.subplots(figsize=(5, 2.5))
matriz_confusao(y_test, prediction_outliers, ax, title="Matriz de Confusão - Dataset com Outliers")
plt.tight_layout()
plt.show()

# Gráfico de Comparação
metricas_original = [accuracy, precision, recall, f1_score]
metricas_noise = [accuracy_noise, precision_noise, recall_noise, f1_noise]
metricas_outliers = [accuracy_outliers, precision_outliers, recall_outliers, f1_outliers]
grafico_metricas(metricas_original, metricas_noise, metricas_outliers)