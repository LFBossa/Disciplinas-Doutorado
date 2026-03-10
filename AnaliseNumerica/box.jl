
# Método Box para a equação da onda
function wave_equation_box(δ, h, k, T_final)
    # Discretização espacial
    x_points = 0:h:π_val
    N = length(x_points) - 1  # número de intervalos
    M = N + 1  # número de pontos
    
    # Discretização temporal
    t_points = 0:k:T_final
    n_steps = length(t_points) - 1
    
    # Inicializar arrays
    U = zeros(M, n_steps+1)
    V = zeros(M, n_steps+1)
    W = zeros(M, n_steps+1)
    
    # Condições iniciais
    for i in 1:M
        x = x_points[i]
        U[i, 1] = u_initial(x)
        V[i, 1] = v_initial(x)
        W[i, 1] = w_initial(x)
    end
    
    # Parâmetro do método
    ν = k / h
    
    # Para cada passo de tempo
    for j in 1:n_steps
        # Montar sistema linear A * U_{j+1} = b
        
        # Número de equações: 
        # 2 equações para cada intervalo (i=1:N) -> 2N
        # + 2 condições de contorno -> total 2(M) = 2(N+1)
        n_equations = 2*M
        A = spzeros(n_equations, n_equations)
        b = zeros(n_equations)
        
        eq_counter = 0
        
        # Condição de contorno em x=0: v(0,t)=0
        eq_counter += 1
        A[eq_counter, 1] = 1.0  # v_0
        b[eq_counter] = 0.0
        
        # Equações para cada intervalo interno (i=1:N)
        for i in 1:N
            # Equação 1 para o intervalo (i, i+1)
            eq_counter += 1
            # Coeficientes para v_i, v_{i+1}, w_i, w_{i+1}
            v_i_idx = 2*(i-1) + 1
            w_i_idx = 2*(i-1) + 2
            v_ip1_idx = 2*i + 1
            w_ip1_idx = 2*i + 2
            
            # v_i + v_{i+1} + ν*w_i - ν*w_{i+1} = R_v
            A[eq_counter, v_i_idx] = 1.0
            A[eq_counter, v_ip1_idx] = 1.0
            A[eq_counter, w_i_idx] = ν
            A[eq_counter, w_ip1_idx] = -ν
            
            # Lado direito R_v
            R_v = V[i, j] + V[i+1, j] + ν*(W[i+1, j] - W[i, j])
            b[eq_counter] = R_v
            
            # Equação 2 para o intervalo (i, i+1)
            eq_counter += 1
            # w_i + w_{i+1} + ν*v_i - ν*v_{i+1} = R_w
            A[eq_counter, w_i_idx] = 1.0
            A[eq_counter, w_ip1_idx] = 1.0
            A[eq_counter, v_i_idx] = ν
            A[eq_counter, v_ip1_idx] = -ν
            
            # Lado direito R_w
            R_w = W[i, j] + W[i+1, j] + ν*(V[i+1, j] - V[i, j])
            b[eq_counter] = R_w
        end
        
        # Condição de contorno em x=π: w_N + δ*v_N = 0
        eq_counter += 1
        v_N_idx = 2*(M-1) + 1
        w_N_idx = 2*(M-1) + 2
        A[eq_counter, v_N_idx] = δ
        A[eq_counter, w_N_idx] = 1.0
        b[eq_counter] = 0.0
        
        # Resolver sistema linear
        U_next = A \ b
        
        # Extrair soluções
        for i in 1:M
            V[i, j+1] = U_next[2*(i-1)+1]
            W[i, j+1] = U_next[2*(i-1)+2]
        end
        
        # Atualizar u usando regra do trapézio
        for i in 1:M
            U[i, j+1] = U[i, j] + (k/2)*(V[i, j] + V[i, j+1])
        end
    end
    
    return x_points, t_points, U, V, W
end



    # Executar simulação
    x_points, t_points, U, V, W = wave_equation_box(δ, h, k, T_final)

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
    plt = plot_solution_at_times(U, x_points, t_points, times_to_plot)
    savefig(plt, "wave_equation_box_method.png")
    display(plt)

    # Análise de estabilidade (número de Courant)
    ν = k/h
    println("Número de Courant ν = $ν")
    println("Estabilidade: O método Box é incondicionalmente estável para sistemas lineares.")