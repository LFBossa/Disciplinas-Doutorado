from random import random
import json
import matplotlib.pyplot as plt
import numpy as np

def funcao_caracteristica(x,y):
    """Define meu conjunto esperado"""
    if abs(x) > 1 or abs(y) > 1:
        return 0
    if abs(y) > 0.8:
        return 1
    raio = x**2 + y**2
    if raio < 0.5 and raio > 0.25:
        return 1
    else:
        return 0
K = 2.5 # dispersão dos dados 
N = 2000
# Dados uniformes
DADOS = []
for i in range(N):
    x = K*(random()-1/2)
    y = K*(random()-1/2)
    z = funcao_caracteristica(x,y)
    DADOS.append([x,y,z])

with open('dados.json', 'w') as f:
    json.dump(DADOS, f, ensure_ascii=False, indent=2)

DADOS = np.array(DADOS) 
# Definindo um mapa de cores com apenas duas cores (por exemplo, azul e vermelho)
from matplotlib.colors import ListedColormap

cmap = ListedColormap(['blue', 'red'])

x_coords = DADOS[:, 0]
y_coords = DADOS[:, 1]
z_coords = DADOS[:, 2]

# Criando o gráfico de dispersão com o novo mapa de cores
plt.figure(figsize=(8, 6))
scatter = plt.scatter(x_coords, y_coords, c=z_coords, cmap=cmap, alpha=0.7, s=10)  # s ajusta o tamanho dos pontos
plt.colorbar(scatter, label='z (cor)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Gráfico de Dispersão dos Dados')
plt.savefig('dados_treino.png', dpi=300)  # Salva o gráfico como PNG com alta resolução 