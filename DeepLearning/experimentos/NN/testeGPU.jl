using CUDA



m = 2000
n = 2000
W = rand(m, n)
x = rand(n)

(cW, cx) = (W, x) |> cu  # move both to GPU
@time cy = cW * cx  # matrix-vector multiplication on GPU
@time y = W * x  # matrix-vector multiplication on CPU



m = 3000
n = 2000
W = rand(m, n)
x = rand(n)

(cW, cx) = (W, x) |> cu  # move both to GPU
@time cy = cW * cx  # matrix-vector multiplication on GPU
@time y = W * x  # matrix-vector multiplication on CPU