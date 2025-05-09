import numpy as np
import pandas as pd
from collections import Counter

# Função euclidiana
def euclidean_fun(x1,x2):
    distance = np.sqrt(np.sum((x1-x2)**2))
    return distance


class KNN:
    # Definir o k
    def __init__(self, k):
        self.k = None if k == 0 else k
    
    # Função de treino (armazena os dados)
    def fit(self, X,y):
        self.X_train = X   # matriz de caracteristicas
        self.y_train = y   # vetor com as classes correspondentes

    # Prevê
    def predict(self,X):
        predictions = [self._predict(x) for x in X]
        return predictions
    
    def _predict(self, x):
        # Calcula as distâncias de cada exemplo ao ponto x
        distances = [euclidean_fun(x, x_train) for x_train in self.X_train]

        # Vai buscar os k valores mais próximos
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # Vê qual é a classe mais comum entre os k escolhidos
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]