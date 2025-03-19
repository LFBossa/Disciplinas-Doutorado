---
title: "Distribuições Multivariadas"
author: "L. F. Bossa"
institute: "UFSC"
theme: "Warsaw"
colortheme: "beaver"
aspectratio: 169
date: 19/03/25
---

# Variável Aleatória em 1 dimensão

$$\mathbb{P}(X \le a) = \int_{-\infty}^{a} f(x) dx $$
com $f$ sendo a função densidade de probabilidade.

## Esperança

Também conhecido como média e denotado por $\mu$.

$$\mathbb{E}(X) = \int_{\mathbb{R}} xf(x) dx$$

---

Em geral, dada $g: \mathbb{R}\to \mathbb{R}$ temos

$$\mathbb{E}(g(X)) = \int_{\mathbb{R}} g(x)f(x) dx$$

---

## Variância

**Variância**

$$\text{Var}(X) = \mathbb{E}( (X - \mu)^2)$$

Mede o quão dispersos estão os dados de $X$: quanto maior a variância, mais longe da média estão os dados.

A raiz quadrada da variância é o desvio-padrão, geralmente denotado por $\sigma$. 


## Variáveis Gaussianas

São caracterizadas pela sua função de densidade

$$f(x) = \frac{1}{\sqrt{2\pi}\sigma}\exp\left(-\frac{(x-\mu)^2}{2\sigma}\right)$$

Uma variável aleatória $X$ com média $\mu$ e variância $\sigma^2$ é denotada por $X\sim N(\mu,\sigma^2)$.

## Z-score

Dada uma variável aleatória X\sim N(\mu,\sigma^2)$, podemos fazer uma transformação 

$$Z = \frac{X- \mu}{\sigma}$$

de modo que $Z \sim  N(0,1)$. 


# Variável Aleatória em várias dimensões


- Temos um vetor $X = (X_1, \ldots, X_n)$ cujas componentes são variáveis aleatórias. 
- Temos uma função de densidade $f(x_1,\ldots, x_n)$ de modo que 

$$\mathbb{P}(X_1 \le a_a, \ldots, X_n \le a_n) = \int_{-\infty}^{a_1}\ldots\int_{-\infty}^{a_n} f(x_1,\ldots,x_n) dx_n\ldots dx_1 $$



## Probabilidade marginal

Surge quando queremos estudar a distribuição geral de uma variável só, "ignorando" as outras. Nesse caso, integramos a variável de interesse no intervalo de interesse, e as outras variáveis são integradas em todo $\mathbb{R}$.

$$\mathbb{P}(X_1 \le a_1)  = \int_{-\infty}^{a_1}\ldots\int_{-\infty}^{\infty} f(x_1,\ldots,x_n) dx_n\ldots dx_1$$

Podemos definir então as densidades marginais

$$f_1(x) =  \int_{-\infty}^{x}\ldots\int_{-\infty}^{\infty} f(x_1,\ldots,x_n) dx_n\ldots dx_1$$


## Esperança

No caso vetorial, o operador esperança atua entrada-a-entrada

$$\mathbb{E}(X) = (\mathbb{E}(X_1), \mathbb{E}(X_2),\ldots, \mathbb{E}(X_n))$$

em que cada esperança é calculada com respeito à distribuição marginal de cada variável.


## Covariância

Dadas variáveis $X$, $Y$ unidimensionais com médias $\mu_X, \mu_Y$, definimos a covariância de $X$ e $Y$ como 
$$\text{Cov}(X,Y) = \mathbb{E}( (X - \mu_X)(Y - \mu_Y))$$

Por vezes também é usada a notação $\sigma_{XY}$ para denotar a covariância.

A covariância indica o quanto $X$ e $Y$ estão _linearmente_ relacionadas.

---

Note que $$\text{Cov}(X,X) = \mathbb{E}( (X - \mu_X)(X - \mu_X)) =  \mathbb{E}((X-\mu_X)^2) =  \text{Var}(X)$$

---

## Independência estatística

Duas variáveis $X_1$ e $X_2$ são ditas independentes se for possível escrever a função de densidade como $f(x_1,x_2) = f_1(x_1)f_2(x_2)$



### Teorema 
Se  $X_1$ e $X_2$  são independentes, então 
$\text{Cov}(X,Y)=0$

---

### Contra-exemplo

A recíproca não é verdadeira. Considere $X$ uma variável aleatória com média zero e densidade $f(x)$ sendo uma função par. Defina $Y = X^2$. 
Nesse caso, claramente $Y$ e $X$ não são independentes, mas 
$$\text{Cov}(X,Y) = 0 $$ 


---

## Correlação

É a covariância "normalizada"

$$\text{Corr}(X,Y) = \frac{\text{Cov}(X,Y)}{\sigma_X\cdot\sigma_Y}$$

Também denotada por $\rho_{XY}$.

## Matriz de Variância-Covariância

- Para calcular a variância de uma variável unidimensional, temos que calcular $\mathbb{E}((X-\mu_X)^2)$
- Por causa do termo quadrático, temos que adaptar esse cálculo para o caso vetorial. 

Sendo $X = (X_1,\ldots,X_n)$ uma variável aleatória e seja $\mathbb{E}(X) = \mu_X$ seu vetor-médio. Calculamos a variância de $X$ como

$$\Sigma = \mathbb{E}((X-\mu_X)(X-\mu_X)^\top)$$

---

$$\Sigma = \mathbb{E}\begin{pmatrix}  (X_1 -\mu_{X_1})(X_1 -\mu_{X_1}) & (X_1 -\mu_{X_1})(X_2 -\mu_{X_2}) \end{pmatrix}$$

---

## Variância total

$$\text{tr}(\Sigma)$$



## Variância Generalizada

$$\text{det}(\Sigma)$$



## Distância Mahalanobis

Dados vetores $X$ e $Y$ que pertençam a uma distribuição multivariada descrita por uma matriz $\Sigma$, definimos a distância Mahalanobi
entre eles como 

$$d_{XY}^2 = (X-Y)^\top\Sigma^{-1}(X-Y)$$

Essa é a generalização do Z-score.

