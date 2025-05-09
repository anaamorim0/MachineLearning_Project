import matplotlib.pyplot as plt
import numpy as np

def matriz_confusao(y_true, y_pred, ax, title):
    unique_labels = np.unique(np.concatenate((y_true, y_pred)))

    classe_negativa, classe_positiva = sorted(unique_labels)

    TP = FP = FN = TN = 0
    for true, pred in zip(y_true, y_pred):
        if true == classe_positiva and pred == classe_positiva:
            TP += 1
        elif true == classe_negativa and pred == classe_positiva:
            FP += 1
        elif true == classe_positiva and pred == classe_negativa:
            FN += 1
        elif true == classe_negativa and pred == classe_negativa:
            TN += 1

    table_data = [
        ["", f"True {classe_positiva}", f"True {classe_negativa}"],
        [f"Predicted {classe_positiva}", f"TP = {TP}", f"FP = {FP}"],
        [f"Predicted {classe_negativa}", f"FN = {FN}", f"TN = {TN}"]
    ]

    table = ax.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.3]*3)
    table.auto_set_font_size(False)
    table.set_fontsize(10)     
    table.scale(1.2, 1.2)        

    table[(1, 1)].set_facecolor('#b3e6b3')
    table[(1, 2)].set_facecolor('#f7c6c7')
    table[(2, 1)].set_facecolor('#f7c6c7')
    table[(2, 2)].set_facecolor('#b3e6b3')

    ax.axis('off')
    ax.set_title(title, fontsize=14, pad=8)
