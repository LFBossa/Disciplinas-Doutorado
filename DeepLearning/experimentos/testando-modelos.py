#%%
import glob 
from random import random
import numpy as np
from gerando_dados import funcao_caracteristica
from sontag import RedeNeural1, RedeNeural2, RedeNeural3
import torch
import json

 

K = 2.5 # dispersão dos dados 
N = 1000
# Dados uniformes
DADOS = []
for i in range(N):
    x = K*(random()-1/2)
    y = K*(random()-1/2)
    z = funcao_caracteristica(x,y)
    DADOS.append([x,y,z])

DADOS = np.array(DADOS) 
 
#%%
if __name__ == "__main__":
    resultados = []
    for sontag in [False, True]:
        for n1 in [32, 128, 256]: 
            model = RedeNeural1(n1, sontag=sontag)   
            model.load_state_dict(torch.load(f'./{model.name}.pth'))
            model.eval()
            entradas = torch.tensor(DADOS[:, :2], dtype=torch.float32)
            saidas = model(entradas).detach().numpy().flatten()
            labels = np.where(saidas >= 0.5, 1.0, 0.0)
            verificacao = labels == DADOS[:, 2]
            erro = (N-verificacao.sum())/N
            resultados.append({'model': model.name, 'erro': erro})

        for n2 in [9, 20, 30]:
            model = RedeNeural2(n2, n2, sontag=sontag) 
            model.load_state_dict(torch.load(f'./{model.name}.pth'))
            model.eval()
            entradas = torch.tensor(DADOS[:, :2], dtype=torch.float32)
            saidas = model(entradas).detach().numpy().flatten()
            labels = np.where(saidas >= 0.5, 1.0, 0.0)
            verificacao = labels == DADOS[:, 2]
            erro = (N-verificacao.sum())/N
            resultados.append({'model': model.name, 'erro': erro})
        for n3 in [7, 15, 21]:
            model = RedeNeural3(n3, n3, n3, sontag=sontag) 
            model.load_state_dict(torch.load(f'./{model.name}.pth'))
            model.eval()
            entradas = torch.tensor(DADOS[:, :2], dtype=torch.float32)
            saidas = model(entradas).detach().numpy().flatten()
            labels = np.where(saidas >= 0.5, 1.0, 0.0)
            verificacao = labels == DADOS[:, 2]
            erro = (N-verificacao.sum())/N
            resultados.append({'model': model.name, 'erro': erro})
    with open('resultados.json', 'w') as f:
         json.dump(resultados, f, indent=2, ensure_ascii=False)
# %% 
