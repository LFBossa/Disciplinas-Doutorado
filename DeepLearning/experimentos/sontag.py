# %% [markdown]
# # Função de Ativação de Sontag
# 
# 
# [Slides Sontag](https://drive.google.com/file/d/1V92dqLg0L3pz5awyvKIavv_FlqtN3Wuz/view)
# 

# %% [markdown]
# 
# #

# %% [markdown]
# ## Referências para treino
# 
# https://www.geeksforgeeks.org/extending-pytorch-with-custom-activation-functions/
# 
# https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html

# %%
import time 
import json
import sys 
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split


# %%
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
# %%

with open('dados.json', 'r') as f:
    DADOS = json.load(f)
    
DADOS = np.array(DADOS) 

# %%
class SigmoideSontag(nn.Module):
    def __init__(self, alpha=100):
        super().__init__()
        self.alpha = alpha
        
    def forward(self, x):
        sigmoid = torch.atan(x)/np.pi + 0.5 + torch.cos(x)/(self.alpha*(1+torch.pow(x,2)))
        return sigmoid


class RedeNeural1(nn.Module):
    def __init__(self, hidden_size=10, sontag=False):
        super(RedeNeural1, self).__init__()
        self.camada1 = nn.Linear(2, hidden_size)  
        self.camada2 = nn.Linear(hidden_size, 1) 
        if sontag:
            self.ativacao = SigmoideSontag()
            self.name = f'Sontag 1 ({hidden_size})'
        else:
            self.ativacao = nn.ReLU() 
            self.name = f'ReLU 1 ({hidden_size})'
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.camada1(x)
        x = self.ativacao(x) 
        x = self.camada2(x)
        x = self.sigmoid(x)
        return x
    
class RedeNeural2(nn.Module):
    def __init__(self, h1=10, h2=10, sontag=False):
        super(RedeNeural2, self).__init__()
        self.camada1 = nn.Linear(2, h1)  
        self.camada2 = nn.Linear(h1, h2)
        self.camada3 = nn.Linear(h2, 1) 
        if sontag:
            self.ativacao = SigmoideSontag()
            self.name = f'Sontag 2 ({h1}, {h2})'
        else:
            self.ativacao = nn.ReLU()
            self.name = f'ReLU 2 ({h1}, {h2})'
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.camada1(x)
        x = self.ativacao(x) 
        x = self.camada2(x)
        x = self.ativacao(x)
        x = self.camada3(x)
        x = self.sigmoid(x)
        return x


class RedeNeural3(nn.Module):
    def __init__(self, h1=10, h2=10, h3=10, sontag=False):
        super(RedeNeural3, self).__init__()
        self.camada1 = nn.Linear(2, h1)  
        self.camada2 = nn.Linear(h1, h2)
        self.camada3 = nn.Linear(h2, h3)
        self.camada4 = nn.Linear(h3, 1) 
        if sontag:
            self.ativacao = SigmoideSontag()
            self.name = f'Sontag 3 ({h1}, {h2}, {h3})'
        else:
            self.ativacao = nn.ReLU()
            self.name = f'ReLU 3 ({h1}, {h2}, {h3})'
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.camada1(x)
        x = self.ativacao(x) 
        x = self.camada2(x)
        x = self.ativacao(x)
        x = self.camada3(x)
        x = self.ativacao(x)
        x = self.camada4(x)
        x = self.sigmoid(x)
        return x

# %%

# Define the Sontag Sigmoid function
# https://www.geeksforgeeks.org/extending-pytorch-with-custom-activation-functions/


# %%
class MeuDataset(Dataset):
    def __init__(self, dados):
        self.dados = dados

    def __len__(self):
        return len(self.dados)

    def __getitem__(self, idx):
        x = torch.tensor(self.dados[idx][:2], dtype=torch.float32)
        y = torch.tensor(self.dados[idx][2], dtype=torch.float32)
        return x, y

# Dividir os dados em treino e teste

train_dataset = MeuDataset(DADOS)
dataset_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)




# %%

def train_model(model, num_epochs=1000): 
    # Define the loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    mean_loss = []
    # Training loop
    for epoch in range(num_epochs):
        for X, y in dataset_loader:
            # Zero the gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(X)
            # Compute the loss
            loss = loss_fn(outputs.squeeze(), y)
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
        mu = loss.item()    
        mean_loss.append(mu)
        if epoch % 100 == 0:
            sys.stdout.write(f'\r{model.name}, Epoch [{epoch+1}/{num_epochs}], Loss: {mu:.4f}')
            sys.stdout.flush()
    return mean_loss

# %%
# Create a grid of points
#x_min, x_max = DADOS[:, 0].min() - 0.5, DADOS[:, 0].max() + 0.5
#y_min, y_max = DADOS[:, 1].min() - 0.5, DADOS[:, 1].max() + 0.5

def plot_figure(model, figure_title, filename):
    x_min, x_max = -1.5, 1.5
    y_min, y_max = -1.5, 1.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                        np.arange(y_min, y_max, 0.01))

    # Prepare the grid points for prediction
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.tensor(grid_points, dtype=torch.float32)

    # Get predictions from the model
    with torch.no_grad():
        predictions = model(grid_tensor).numpy()

    # Reshape predictions to match the grid shape
    predictions = predictions.reshape(xx.shape)

    # Plot the decision boundary
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, predictions, levels=50, cmap='coolwarm', alpha=0.8)
    plt.colorbar(label='Prediction Probability')

    # Overlay the original data points
    #plt.scatter(DADOS[:, 0], DADOS[:, 1], c=DADOS[:, 2], cmap=cmap, edgecolor='k', s=7)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(figure_title)
    plt.savefig(filename)
    plt.close()
#plt.show()


def train_loop(model, num_epochs): 
    start = time.time()
    mean_loss = train_model(model, num_epochs)
    end = time.time()
    print(f"\nTempo de treinamento para {model.name}: {end - start:.2f} segundos")
    plot_figure(model, f'{model.name}', f'{model.name}.png')
    return {'name': model.name, 'time': end - start, 'loss': mean_loss, 'num_epochs': len(mean_loss)}

if __name__ == "__main__":
    # Treinando a rede
    num_epochs = 2000
    metadados_treino = [] 
    for sontag in [False, True]:
        for n1 in [32, 128, 256]: 
            model = RedeNeural1(n1, sontag=sontag) 
            metadados = train_loop(model, num_epochs)
            metadados_treino.append(metadados)
            torch.save(model.state_dict(), f'{model.name}.pth')
        for n2 in [9, 20, 30]:
            model = RedeNeural2(n2, n2, sontag=sontag)
            metadados = train_loop(model, num_epochs)
            metadados_treino.append(metadados)
            torch.save(model.state_dict(), f'{model.name}.pth')
        for n3 in [7, 15, 21]:
            model = RedeNeural3(n3, n3, n3, sontag=sontag)
            metadados = train_loop(model, num_epochs)
            metadados_treino.append(metadados)
            torch.save(model.state_dict(), f'{model.name}.pth')
    with open('metadados_treino.json', 'w') as f:
        json.dump(metadados_treino, f, ensure_ascii=False, indent=2)


# %%
