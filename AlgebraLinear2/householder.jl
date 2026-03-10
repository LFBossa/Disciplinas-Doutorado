using LinearAlgebra
using SparseArrays
using Plots
using BlockDiagonals
using MAT

function Householder(v)
    """Retorna o vetor u tal que H = I - 2vv' é a transformação que leva
    v em ||v||e_n"""
    n = length(v)
    σ = v[1] >= 0 ? -1.0 : 1.0
    u = copy(v)
    u[1] -= σ * norm(v)
    H = I(n) - 2 * u * u' / (u'u)
    return H
end

function HessenberbyHouseholder(A)
    """Retorna uma matrix H tal que HAH é Hessenberg."""
    m, n = size(A)
    Gis = []
    B = copy(A)
    for j = 1:n-2
        v = B[j+1:m, j]
        Hj = Householder(v)
        Gj = BlockDiagonal([1.0 * I(j), Hj])
        push!(Gis, Gj)
        B = Gj * B * Gj'
    end
    return *(reverse(Gis)...) # Gn*...*G1
end
function givens_rotation(a, b)
    if b == 0
        return 1.0, 0.0
    else
        r = hypot(a, b)
        c = a / r
        s = -b / r
        return c, s
    end
end 

function QRGivens(A)
    m, n = size(A)
    B = copy(A)
    Q_product = I(m) # Initialize the product of Givens matrices
    for j in 1:n-1
        c, s = givens_rotation(B[j, j], B[j+1, j])
        # Create Givens matrix
        G = BlockDiagonal([1.0 * I(j-1), [c -s; s c], 1.0 * I(m-j-1)]) 
        Q_product = G * Q_product # Update the product of Givens matrices
    end
    return Q_product, B
end
function IteracaoSubespacos(A, k, nmax=1000, tol=1e-10)
  m, n = size(A)
  Qtil = randn(n,k)
  Q, R = qr(Qtil)
  Q = Q[:,1:k]
  for j = 1:nmax
    Z = A*Q
    Q, R = qr(Z)
    Q = Q[:,1:k]
    T = Q'A*Q
    ϵ = norm(T - UpperTriangular(T), Inf)
    if ϵ < tol # se atingimos a tolerância, paramos
     break
    end
  end
  return Q
end

function IteracaoSubespacosInversa(A, k, nmax=100, tol=1e-9)
  m, n = size(A)
  Qtil = rand(n,k)
  Q, R = qr(Qtil)
  Q = Q[:,1:k]
  j = 0
  for j = 1:nmax
    Z = A\Q
    Q, R = qr(Z)
    Q = Q[:,1:k]
    T = Q'A*Q
    ϵ = norm(T - UpperTriangular(T))
    if ϵ < tol
      println(j)
      break
    end
  end
  return Q
end
function Vandermonde(x, p)
  n = length(x)
  V = zeros(ComplexF64, p+1,n)
  for i=0:p
    V[i+1,:] = x.^i
  end
  V
end
 

vars = matread("sinalh.mat");
h = vars["sinalh"][:,1];
M,  = size(h)
# Vamos usar 100 pontos

function  ReconstruirSinal0(h, N) 
    M = length(h)
    H0 = hcat([h[1+k:N+k] for k=0:N-1]...)
    H1 = hcat([h[1+k:N+k] for k=1:N]...);
    f = (H0'H0)\(H0'H1[:,end]);
    C = hcat([ zeros(1,N-1);
                I(N-1)], f)
    n = rank(H0)

    Q = IteracaoSubespacos(H0, n, 10000) 

    z = eigen(Q'C*Q).values
    
        plot(real(z), imag(z), label="λ(C)", seriestype = :scatter, xlabel = "Re(λ)", ylabel = "Im(λ)", title = "Eigenvalues of Q'CQ")

        # Add unit circle
        theta = range(0, stop=2*pi, length=100)
        plot!(cos.(theta), sin.(theta), label = "Unit Circle")

        # Set axis limits to be the same
        max_val = max(maximum(abs.(real(z))), maximum(abs.(imag(z))))
        plot!(xlims=(-max_val - 0.2, max_val + 0.2), ylims=(-max_val - 0.2, max_val + 0.2), aspect_ratio = :equal)
    
    V = Vandermonde(z,M-1);
    r = V\h
    Λ = diagm(r);
    plot(real((V*Λ*V')[:,1]), label="Reconstruido, N=$N")
    plot!(h, label="Original")
                        
end

ReconstruirSinal0(h, 40)

A = [2 4 5 6  0;
  -1.0 0 2 3 -2;
     2 3 4 -2 1/4;
     34 4 -1 0 2;
     0 2 3 4 5]

Q = IteracaoSubespacos(A, 3, 10000, 1e-10);
Q'A*Q

htilde = vars["sinalh"][:,2];

ReconstruirSinal0(htilde, 30)