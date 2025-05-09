import matplotlib.pyplot as plt
import numpy as np

# Grafico comparativo para cada métrica das diferentes databases
def grafico_metricas(metricas, metricas_noise, metricas_outliers, nomes_metricas=None):
    if nomes_metricas is None:
        nomes_metricas = ["Accuracy", "Precision", "Recall", "F1-Score"]

    x = np.arange(len(nomes_metricas)) 
    width = 0.25  # Largura de cada barra

    fig, ax = plt.subplots(figsize=(10, 6))

    rects1 = ax.bar(x - width, metricas, width, label='Dataset Original')
    rects2 = ax.bar(x, metricas_noise, width, label='Dataset com Noise')
    rects3 = ax.bar(x + width, metricas_outliers, width, label='Dataset com Outliers')

    ax.set_ylabel('Valor das Métricas')
    ax.set_title('Gráfico de Comparação das Métricas de Avaliação para os Datasets')
    ax.set_xticks(x)
    ax.set_xticklabels(nomes_metricas)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.tight_layout()
    plt.show()