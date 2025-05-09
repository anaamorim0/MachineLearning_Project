import numpy as np

# cria outliers
def outliers(X, y, nivel_outliers, magnitude=10.0): # magnitude = até quanto queremos "sair" dos limites normais para gerar outliers
    n_outliers = int(len(X) * nivel_outliers)   # calcula quantos outliers vão ser criados
    n_features = X.shape[1] # conta o n de atributos que cada exemplo tem
    
    # define um intervalo de valores extremo para gerar os outliers
    # usa o 5º percentil como limite inferior
    low = np.percentile(X, 5) - magnitude
    # usa o 95º percentil como limite superior
    high = np.percentile(X, 95) + magnitude

    # gera n_outliers exemplos aleatorios com valores entre o low e o high
    outlier_X = np.random.uniform(
        low=low,
        high=high,
        size=(n_outliers, n_features)   # cada outlier tem n_features atribuidos
    )

    unique_labels = np.unique(y)    # encontra todos os valores diferentes que existem em y (classes)
    outlier_y = np.random.choice(unique_labels, size=n_outliers)    # escolhe aleatoriamente uma classe para cada um dos novos outliers
    
    # junta os novos outliers ao dataset original
    new_X = np.vstack((X, outlier_X))
    new_y = np.hstack((y, outlier_y))
    
    return new_X, new_y