using StaticArrays
function doit(mu::SVector, V::SMatrix, n)
    L = cholesky(V).U'
    X = [normalize(mu + L * @SVector(randn(3))) for _=1:n]
    return X
end

mu_s = SA[2., 2., 2.]
V_s = SA[0.74 -0.08 0.34; -0.08 1.57 -0.5; 0.34 -0.5 1.16]
doit($mu_s, $V_s, $n);