### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ b7ea71bf-3bfd-4ef4-ae16-339b52fb3b09
using LinearAlgebra

# ╔═╡ ebdda970-fec7-11ef-04b5-e5b54dc295da
md"""
# Aula 01 - Normas

Seja $x = (x_1,x_2,\ldots,x_n)$ um vetor em $\mathbb{R}^n$

Normas estudadas:
- Norma euclidiana (norma-2)
$$\|x\|_2 = \sqrt{\sum_{i=1}^n x_i^2}$$
- Norma 1
$$\|x\|_1 = \sum_{i=1}^n |x_i|$$
- Norma infinito
$$\|x\|_{\infty} = \max_{i=1,\ldots,n} |x_i|$$
- Norma-$p$ (generalização de todas as acima)
$$\|x\|_p = \left(\sum_{i=1}^n x_i^p\right)^{1/p}$$

- Norma energia: dada uma matriz $A$ simétrica positiva definida, ela induz uma norma dada por 

$$\|x\|_A = \sqrt{x^\top Ax}$$
"""

# ╔═╡ 5d0ad130-0eed-4db1-b8e0-4c461b67b016
md"Em `julia`, podemos importar o pacote `LinearAlgebra` e temos a função `norm()`"

# ╔═╡ f9a84693-0340-45c7-907a-c3c7e7bdbeba
x = [3,-4,5]

# ╔═╡ 30605be8-269c-4c2b-91db-0cd397888009
norm(x,1)

# ╔═╡ ce7469f5-2e11-493a-836f-3987e005e636
norm(x,2)

# ╔═╡ 23700332-7331-49f0-9f03-af9deef11229
norm(x,Inf)

# ╔═╡ 2960b63a-0aeb-4ed8-b330-13aeefc61669
A = [2 0 1; 0 1 0; 1 0 3]

# ╔═╡ 9913dbb4-7d7a-4c7e-b7c5-2dee968b95f7
# usamos apóstrofo para transpor
sqrt(x'*A*x)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.4"
manifest_format = "2.0"
project_hash = "ac1187e548c6ab173ac57d4e72da1620216bce54"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.11.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.27+1"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"
"""

# ╔═╡ Cell order:
# ╟─ebdda970-fec7-11ef-04b5-e5b54dc295da
# ╠═b7ea71bf-3bfd-4ef4-ae16-339b52fb3b09
# ╟─5d0ad130-0eed-4db1-b8e0-4c461b67b016
# ╠═f9a84693-0340-45c7-907a-c3c7e7bdbeba
# ╠═30605be8-269c-4c2b-91db-0cd397888009
# ╠═ce7469f5-2e11-493a-836f-3987e005e636
# ╠═23700332-7331-49f0-9f03-af9deef11229
# ╠═2960b63a-0aeb-4ed8-b330-13aeefc61669
# ╠═9913dbb4-7d7a-4c7e-b7c5-2dee968b95f7
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
