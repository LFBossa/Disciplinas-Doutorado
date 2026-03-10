function [alpha_out, beta_out, U, V] = GBK2(A, b, k, ortog)
% GBK2 Realiza k passos da bidiagonalização de Golub-Kahan
%   [alpha, beta, U, V] = GBK2(A, b, k, ortog)
%
%   Entradas:
%     A     - Matriz do sistema
%     b     - Vetor lado direito
%     k     - Número de passos
%     ortog - (Opcional) Booleano para realizar re-ortogonalização (padrão: false)

  if nargin < 4
    ortog = false;
  end

  tol = 1e-14;
  [m, n] = size(A);

  % Garante que b é um vetor coluna
  b = b(:);

  % Inicializando
  beta1 = norm(b);
  u1 = b / beta1;
  r1 = A' * u1;
  alpha1 = norm(r1);
  v1 = r1 / alpha1;

  alpha_vec = [alpha1];
  beta_vec = [beta1];
  U = [u1];
  V = [v1];

  for j = 1:k
    % Acessando a j-ésima coluna de V e U
    v_j = V(:, j);
    u_j = U(:, j);
    
    pj1 = A * v_j - alpha_vec(j) * u_j;
    bj1 = norm(pj1);

    if bj1 < tol
      break;
    else
      beta_vec = [beta_vec; bj1]; % push!(beta, bj1)
      uj1 = pj1 / bj1;
      
      if ortog
        uj1 = ortogonalizar(U, uj1);
        % Nota: Em implementações robustas, muitas vezes re-normaliza-se aqui
      end
      
      U = [U, uj1]; % push!(U, uj1)
      
      rj1 = A' * uj1 - bj1 * v_j;
      aj1 = norm(rj1);
      
      if aj1 < tol
        break;
      else
        alpha_vec = [alpha_vec; aj1]; % push!(alpha, aj1)
        vj1 = rj1 / aj1;
        
        if ortog
          vj1 = ortogonalizar(V, vj1);
        end
        
        V = [V, vj1]; % push!(V, vj1)
      end
    end
  end

  % Retorna vetores alfa e beta e matrizes U e V
  % Em Julia: beta[2:end] remove o primeiro elemento (beta1)
  alpha_out = alpha_vec;
  if length(beta_vec) > 1
      beta_out = beta_vec(2:end);
  else
      beta_out = [];
  end
end

function v_out = ortogonalizar(Q, v)
  % Ortogonalização via Gram-Schmidt Modificado
  % Remove de v as projeções nas colunas de Q
  v_out = v;
  for i = 1:size(Q, 2)
    qi = Q(:, i);
    v_out = v_out - (qi' * v_out) * qi;
  end
end