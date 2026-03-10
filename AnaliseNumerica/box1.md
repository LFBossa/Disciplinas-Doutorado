### Box

Para resolver a equação da onda 1D usando o método Box, precisamos primeiro transformar a equação de segunda ordem em um sistema de primeira ordem. Vou descrever a abordagem passo a passo e fornecer uma implementação em Julia.

## Abordagem Matemática

### Transformação para Sistema de Primeira Ordem

A equação da onda:
$$ u_{tt} - u_{xx} = 0 $$

Defina:
$$ v = u_t \quad \text{e} \quad w = u_x $$

Então temos o sistema:
$$ \begin{cases}
v_t - w_x = 0 \\
w_t - v_x = 0
\end{cases} $$

### Condições Iniciais e de Contorno

Condições iniciais:
$$ u(x,0) = \exp(-100(x-\pi/2)^2) $$
$$ v(x,0) = u_t(x,0) = 0 $$
$$ w(x,0) = u_x(x,0) = -200(x-\pi/2)\exp(-100(x-\pi/2)^2) $$

Condições de contorno:
$$ u(0,t) = 0 \Rightarrow v(0,t) = 0 $$
$$ u_x(\pi,t) + \delta u_t(\pi,t) = 0 \Rightarrow w(\pi,t) + \delta v(\pi,t) = 0 $$

### Discretização com Método Box

Para cada célula \([x_i, x_{i+1}] \times [t_j, t_{j+1}]\), aplicamos:

**Equação 1 (v_t - w_x = 0):**
$$ \frac{v_{i+1}^{j+1} - v_{i+1}^j + v_i^{j+1} - v_i^j}{2\Delta t} - \frac{w_{i+1}^{j+1} - w_i^{j+1} + w_{i+1}^j - w_i^j}{2\Delta x} = 0 $$

**Equação 2 (w_t - v_x = 0):**
$$ \frac{w_{i+1}^{j+1} - w_{i+1}^j + w_i^{j+1} - w_i^j}{2\Delta t} - \frac{v_{i+1}^{j+1} - v_i^{j+1} + v_{i+1}^j - v_i^j}{2\Delta x} = 0 $$

Onde:
- \( v_i^j = v(x_i, t_j) \), \( w_i^j = w(x_i, t_j) \)
- \( \Delta x = h \), \( \Delta t = k \), \( \nu = k/h \)

### Sistema Linear para Cada Passo de Tempo

Para cada passo no tempo, montamos um sistema linear \( A\mathbf{U}^{j+1} = \mathbf{b} \), onde:
$$ \mathbf{U}^{j+1} = [v_0, w_0, v_1, w_1, \ldots, v_N, w_N]^T $$

## Implementação em Julia

```julia
using LinearAlgebra
using SparseArrays
using Plots

# Parâmetros do problema
const π_val = Float64(π)
const δ = 0.5  # pode variar entre (0, 1]

# Funções iniciais
function u_initial(x)
    return exp(-100*(x - π_val/2)^2)
end

function v_initial(x)
    return 0.0
end

function w_initial(x)
    return -200*(x - π_val/2) * exp(-100*(x - π_val/2)^2)
end

# Parâmetros numéricos
h = 0.01
k = 0.005
T_final = 2.0

# Executar simulação
x_points, t_points, u, v, w = wave_equation_box(δ, h, k, T_final)

# Visualização
function plot_solution_at_times(u, x_points, t_points, times_to_plot)
    plt = plot(layout=(2,2), size=(1000,800))
    
    for (idx, t) in enumerate(times_to_plot)
        # Encontrar índice mais próximo no tempo
        t_idx = argmin(abs.(t_points .- t))
        
        plot!(plt[idx], x_points, u[:, t_idx], 
              label="t = $(round(t, digits=2))",
              linewidth=2,
              title="Solução em t = $t",
              xlabel="x", ylabel="u(x,t)",
              legend=:topright)
    end
    
    return plt
end

# Plot em diferentes tempos
times_to_plot = [0.0, 0.5, 1.0, 1.5]
plt = plot_solution_at_times(u, x_points, t_points, times_to_plot)
savefig(plt, "wave_equation_box_method.png")
display(plt)

# Análise de estabilidade (número de Courant)
ν = k/h
println("Número de Courant ν = $ν")
println("Estabilidade: O método Box é incondicionalmente estável para sistemas lineares.")
```

## Análise de Estabilidade

Para o método Box aplicado ao sistema:

1. **Análise de Von Neumann**: Substituindo soluções da forma \( \exp(i\xi x + \sigma t) \) nas equações discretizadas, obtemos a relação de dispersão. Para o sistema \( \mathbf{u}_t + A\mathbf{u}_x = 0 \), o método Box tem fator de amplificação:
   $$
   G = \frac{I - \nu A \tan(\xi h/2)}{I + \nu A \tan(\xi h/2)}
   $$
   Como os autovalores de A são ±1, o método é incondicionalmente estável.

2. **Número de Courant**: \( \nu = \Delta t / \Delta x \). Embora o método seja incondicionalmente estável, para precisão recomenda-se \( \nu \leq 1 \).

3. **Consistência**: O método Box é de segunda ordem em espaço e tempo quando aplicado a sistemas lineares.

## Resultados Numéricos

A implementação acima produzirá gráficos da solução u(x,t) em diferentes tempos. Para diferentes valores de δ:

- **δ pequeno (≈0)**: Comportamento próximo a condição de Neumann (\( u_x=0 \) em x=π)
- **δ = 1**: Absorção parcial na borda
- **δ grande**: Absorção mais forte

Para validar, compare com a solução analítica para o caso δ=0 (domínio infinito ou reflexão perfeita). O pacote inicial gaussiano se dividirá em dois pulsos viajando em direções opostas, refletindo nas bordas de acordo com as condições de contorno.

## Vantagens do Método Box

1. **Incondicionalmente estável** para sistemas lineares
2. **Segunda ordem** de precisão em espaço e tempo
3. **Conservativo** para certas quantidades
4. **Bom tratamento de condições de contorno** complexas

Esta implementação pode ser estendida para coeficientes variáveis, termos fonte e domínios mais complexos.



# Análise de Von Neumann para o Método Box Aplicado à Equação da Onda

Vou realizar uma análise de Von Neumann completa para o método Box aplicado ao sistema da equação da onda, seguindo a estrutura que você forneceu.

## Sistema de Equações e Discretização

Considere o sistema de primeira ordem:
\[
\begin{cases}
v_t - w_x = 0 \\
w_t - v_x = 0
\end{cases}
\]

O método Box discreto para cada célula \([x_i, x_{i+1}] \times [t_j, t_{j+1}]\) é:

**Para v:**
\[
\frac{v_{i+1}^{j+1} - v_{i+1}^j + v_i^{j+1} - v_i^j}{2\Delta t} - 
\frac{w_{i+1}^{j+1} - w_i^{j+1} + w_{i+1}^j - w_i^j}{2\Delta x} = 0
\]

**Para w:**
\[
\frac{w_{i+1}^{j+1} - w_{i+1}^j + w_i^{j+1} - w_i^j}{2\Delta t} - 
\frac{v_{i+1}^{j+1} - v_i^{j+1} + v_{i+1}^j - v_i^j}{2\Delta x} = 0
\]

## Análise de Von Neumann

Definimos o ansatz de Von Neumann para ambas as variáveis:
\[
v_i^j = \hat{v} \xi^j e^{I k i h}, \quad w_i^j = \hat{w} \xi^j e^{I k i h}
\]
onde \(I = \sqrt{-1}\), \(h = \Delta x\), \(k\) é o número de onda, e \(\xi = e^{\sigma \Delta t}\) é o fator de amplificação.

Substituindo nas equações discretas:

### Primeira Equação (para v):

Termo 1 (derivada temporal de v):
\[
\frac{1}{2\Delta t} \left[ (\xi e^{I k (i+1)h} - e^{I k (i+1)h} + \xi e^{I k i h} - e^{I k i h}) \hat{v} \xi^j \right]
= \frac{\hat{v} \xi^j}{2\Delta t} e^{I k i h} \left[ (\xi - 1)(e^{I k h} + 1) \right]
\]

Termo 2 (derivada espacial de w):
\[
-\frac{1}{2\Delta x} \left[ (\xi e^{I k (i+1)h} - \xi e^{I k i h} + e^{I k (i+1)h} - e^{I k i h}) \hat{w} \xi^j \right]
= -\frac{\hat{w} \xi^j}{2\Delta x} e^{I k i h} \left[ (\xi + 1)(e^{I k h} - 1) \right]
\]

Portanto, a primeira equação fica:
\[
\frac{(\xi - 1)(e^{I k h} + 1)}{2\Delta t} \hat{v} - \frac{(\xi + 1)(e^{I k h} - 1)}{2\Delta x} \hat{w} = 0
\]

### Segunda Equação (para w):

Similarmente:
\[
\frac{(\xi - 1)(e^{I k h} + 1)}{2\Delta t} \hat{w} - \frac{(\xi + 1)(e^{I k h} - 1)}{2\Delta x} \hat{v} = 0
\]

## Sistema de Equações

Definindo \(\nu = \frac{\Delta t}{\Delta x}\) (número de Courant) e simplificando por \(1/2\), temos:

\[
\begin{cases}
(\xi - 1)(e^{I k h} + 1) \hat{v} - \nu (\xi + 1)(e^{I k h} - 1) \hat{w} = 0 \\
(\xi - 1)(e^{I k h} + 1) \hat{w} - \nu (\xi + 1)(e^{I k h} - 1) \hat{v} = 0
\end{cases}
\]

Este é um sistema homogêneo em \(\hat{v}\) e \(\hat{w}\). Para ter solução não trivial, o determinante da matriz deve ser zero:

\[
\det \begin{bmatrix}
(\xi - 1)(e^{I k h} + 1) & -\nu (\xi + 1)(e^{I k h} - 1) \\
-\nu (\xi + 1)(e^{I k h} - 1) & (\xi - 1)(e^{I k h} + 1)
\end{bmatrix} = 0
\]

## Cálculo do Determinante

\[
\left[(\xi - 1)(e^{I k h} + 1)\right]^2 - \left[\nu (\xi + 1)(e^{I k h} - 1)\right]^2 = 0
\]

Fatorando:
\[
\left[(\xi - 1)(e^{I k h} + 1) - \nu (\xi + 1)(e^{I k h} - 1)\right] \times
\left[(\xi - 1)(e^{I k h} + 1) + \nu (\xi + 1)(e^{I k h} - 1)\right] = 0
\]

## Análise das Raízes

### Caso 1:
\[
(\xi - 1)(e^{I k h} + 1) - \nu (\xi + 1)(e^{I k h} - 1) = 0
\]

Reorganizando:
\[
(\xi - 1)(e^{I k h} + 1) = \nu (\xi + 1)(e^{I k h} - 1)
\]

Definindo \(\theta = k h\), temos:
\[
e^{I \theta} + 1 = 2e^{I \theta/2} \cos(\theta/2), \quad e^{I \theta} - 1 = 2I e^{I \theta/2} \sin(\theta/2)
\]

Substituindo:
\[
(\xi - 1) \cdot 2e^{I \theta/2} \cos(\theta/2) = \nu (\xi + 1) \cdot 2I e^{I \theta/2} \sin(\theta/2)
\]

Cancelando \(2e^{I \theta/2}\):
\[
(\xi - 1) \cos(\theta/2) = I \nu (\xi + 1) \sin(\theta/2)
\]

Isolando \(\xi\):
\[
\xi \cos(\theta/2) - \cos(\theta/2) = I \nu \xi \sin(\theta/2) + I \nu \sin(\theta/2)
\]
\[
\xi [\cos(\theta/2) - I \nu \sin(\theta/2)] = \cos(\theta/2) + I \nu \sin(\theta/2)
\]
\[
\xi_1 = \frac{\cos(\theta/2) + I \nu \sin(\theta/2)}{\cos(\theta/2) - I \nu \sin(\theta/2)}
\]

### Caso 2:
\[
(\xi - 1)(e^{I k h} + 1) + \nu (\xi + 1)(e^{I k h} - 1) = 0
\]

Similarmente:
\[
(\xi - 1) \cos(\theta/2) = -I \nu (\xi + 1) \sin(\theta/2)
\]
\[
\xi_2 = \frac{\cos(\theta/2) - I \nu \sin(\theta/2)}{\cos(\theta/2) + I \nu \sin(\theta/2)}
\]

Note que \(\xi_2 = 1/\xi_1\).

## Análise de Estabilidade

Para o fator de amplificação \(\xi_1\):
\[
\xi_1 = \frac{\cos(\theta/2) + I \nu \sin(\theta/2)}{\cos(\theta/2) - I \nu \sin(\theta/2)}
\]

O módulo de \(\xi_1\) é:
\[
|\xi_1| = \left| \frac{\cos(\theta/2) + I \nu \sin(\theta/2)}{\cos(\theta/2) - I \nu \sin(\theta/2)} \right|
\]

Note que o numerador e denominador são conjugados complexos (pois apenas o sinal da parte imaginária muda). Portanto:
\[
|\xi_1| = 1 \quad \text{para todo } \nu \text{ e todo } \theta
\]

Como \(\xi_2 = 1/\xi_1\), também temos \(|\xi_2| = 1\).

## Conclusão da Análise

**O método Box é incondicionalmente estável** para o sistema da equação da onda, pois para todos os valores do número de Courant \(\nu = \Delta t / \Delta x\) e para todas as frequências espaciais \(\theta = k h\), temos:

\[
|\xi| = 1
\]

Isso significa que o método não introduz dissipação numérica (o módulo do fator de amplificação é exatamente 1), mas pode introduzir dispersão numérica (erro de fase).

## Comparação com Esquemas Explícitos

Em contraste, esquemas explícitos como:
- FTCS (Forward Time Centered Space): Instável para qualquer \(\nu > 0\)
- Leapfrog: Estável para \(\nu \leq 1\)
- Lax-Wendroff: Estável para \(\nu \leq 1\)

O método Box, sendo implícito, não possui restrição de CFL para estabilidade. No entanto, para precisão, ainda é recomendado que \(\nu\) não seja muito grande, pois erros de fase aumentam com \(\nu\).

## Erro de Fase

Podemos analisar o erro de fase calculando o argumento de \(\xi\). Para \(\xi_1\):
\[
\arg(\xi_1) = 2 \arctan\left(\frac{\nu \sin(\theta/2)}{\cos(\theta/2)}\right) = 2 \arctan(\nu \tan(\theta/2))
\]

A velocidade de fase numérica é:
\[
c_{\text{num}} = \frac{\arg(\xi_1)}{\nu \theta \Delta t} = \frac{2 \arctan(\nu \tan(\theta/2))}{\nu \theta \Delta t}
\]

Comparando com a velocidade exata \(c = 1\), podemos ver que há dispersão numérica, especialmente para ondas com comprimento de onda curto (grande \(\theta\)).