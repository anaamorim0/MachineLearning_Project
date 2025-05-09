import numpy as np

# cálculo da accuracy, precision, recall e F1-score
def metricas(y_true, y_predicted):
    unique_labels = np.unique(np.concatenate((y_true, y_predicted)))    # junta as classes true e predicted e encontra todas 

    classe_negativa, classe_positiva = sorted(unique_labels)  # organiza por ordem crescente qual é a classe negativa e a positiva

    TP = FP = FN = TN = 0   # inicializa os contadores
    for true, predicted in zip(y_true, y_predicted):
        if true == classe_positiva and predicted == classe_positiva:
            TP += 1
        elif true == classe_negativa and predicted == classe_positiva:
            FP += 1
        elif true == classe_positiva and predicted == classe_negativa:
            FN += 1
        elif true == classe_negativa and predicted == classe_negativa:
            TN += 1

    accuracy  = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score  = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return accuracy, precision, recall, f1_score
