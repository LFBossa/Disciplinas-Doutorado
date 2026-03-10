function GBK2(A, bp::Vector, b::Vector, k::Number ; ortog=false)
  # realiza k passos da bidiagonalização
  tol = 1e-14
  δ = norm(bp - b)
  τ = 1.1
  m, n = size(A)

  # Inicializando 
  β1 = norm(b)
  u1 = b/β1
  r1 = A'u1
  α1 = norm(r1)
  v1 = r1/α1

  α = [ α1 ]
  β = [ β1 ]
  U = [ u1 ]
  V = [ v1 ]
  for j = 1:k
    pj1 = A*V[j] - α[j]*U[j]
    bj1 = norm(pj1)
    if bj1 < tol
      break
    else
      push!(β, bj1) # atualiza b_{j+1}
      uj1 = pj1/bj1
      if ortog == true
        uj1 = ortogonalizar(U,uj1)
      end
      push!(U, uj1) # atualiza u_{j+1}
      rj1 = A'uj1 - bj1*V[j]
      aj1 = norm(rj1)
      if aj1 < tol
        break
      else
        push!(α, aj1) # atualiza a_{j+1}
        vj1 = rj1/aj1
        if ortog == true
          vj = ortogonalizar(V,vj1)
        end
        push!(V,vj1) # atualiza v_{j+1}
      end
    end
    
    Vk = hcat(V...)
    AVk = A*Vk
    yk = pinv(AVk)*bp
    xk = Vk*yk
    erro = norm(A*xk-bp)
    if erro ≤ τ*δ
      @info "O critério de discrepância foi atingido com k=$k"
      return xk
    end
  end
  # retorna vetores alfa e beta e matrizes U e V
  #return α, β[2:end], hcat(U...), hcat(V...)
end