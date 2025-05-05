import matplotlib.pyplot as plt
import json
import numpy as np

with open(f"metadados_treino.json", "r") as f:
    metadados = json.load(f)

# Iterar sobre cada entrada nos metadados e agrupar os loss em lotes de 100
for entry in metadados:
    model_name = entry['name']
    training_time = entry['time']
    loss_values = entry['loss']
    
    # Dividir os loss em lotes de 100
    batches = [loss_values[i:i + 100] for i in range(0, len(loss_values), 100)]
    batch_labels = [f"{i * 100}" for i in range(len(batches))]
    
    # Criar o boxplot dos lotes
    plt.figure(figsize=(10, 6))
    plt.boxplot(batches, labels=batch_labels, patch_artist=True)
    plt.xlabel('Intervalo de Épocas')
    plt.ylabel('Loss')
    plt.suptitle(f'{model_name}', fontsize=14, fontweight='bold')
    plt.title(f'Tempo de Treino: {training_time:.2f}s')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    
    # Salvar o gráfico como imagem
    plt.savefig(f"{model_name} loss.png")
    plt.close() 