import numpy as np

# adiciona noise ás labels(y), ou seja, altera valores de classe para valores incorretos
def noise(y, nivel_noise):
    y_noisy = y.copy()
    n = int(nivel_noise * len(y)) # calculo de quantos elementos do vetor vão ser alterados conforme o nível de noise que se introduz
    indices = np.random.choice(len(y), n, replace=False) # escolhe aleatoriamente n linhas diferentes para alterar (replace = False impede repetições) 
    for i in indices:   # para cada indice
        current = y_noisy[i]    # guarda a classe atual
        other_classes = list(set(y_noisy) - {current})  # cria uma lista de todas as classes diferentes da atual
        if other_classes:   # se a lista não for nula
            y_noisy[i] = np.random.choice(other_classes)   # altera para uma outra classe
    return y_noisy