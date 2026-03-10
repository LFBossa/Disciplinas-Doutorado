### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ 58d9bc5c-c7d2-11f0-3092-af17b39f1914
md"# AN Trabalho final"

# ╔═╡ 439bbc96-471d-4788-87a1-af1dfd62b932
md"""**5.38. Resolver a equação:**
$$u_t + a(x, t)u_x = 0, \quad x \ge 0, \quad t \ge 0,$$


onde
$$a(x,t) = \frac{1 + x^2}{1 + 2xt + 2x^2 + x^4},$$
com a condição inicial:

**a.**
$$u(x, 0) = \begin{cases} 1 & 0.2 \le x \le 0.4 \\ 0 & \text{caso contrário} \end{cases}$$
$$u(0, t) = 0,$$
cuja solução exata é $u(x,t) = u\left(\frac{x - t}{1 + x^2}, 0\right)$.

**b.** $$u(x, 0) = \exp(-10(4x - 1)^2)$$ e $$u(0, t) = 0.$$
Usar malhas com $h = 0.02$ e $\Delta t = 0.01$.

Compare os perfis das soluções em $t = 0$, $t = 0.1$, $t = 0.5$ e $t = 1$, obtidas pelos métodos
- **upwind**,
- **Lax-Wendroff**,
- **método Box**
- **esquema implícito de primeira ordem**.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.4"
manifest_format = "2.0"
project_hash = "da39a3ee5e6b4b0d3255bfef95601890afd80709"

[deps]
"""

# ╔═╡ Cell order:
# ╟─58d9bc5c-c7d2-11f0-3092-af17b39f1914
# ╟─439bbc96-471d-4788-87a1-af1dfd62b932
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
