using Flux, Distributions, LinearAlgebra, Statistics, Plots

function initialize_weights(input_size, output_size)
    # Inicializa pesos com distribuição multivariada
    # Média zero e variância total 1 por camada
    scale = 1 / sqrt(input_size)  # Escala para manter variância estável
    W = randn(output_size, input_size) * scale
    b = randn(output_size) * scale
    return (W, b)
end

function create_network()
    # Cria rede neural com 3 camadas: 3 → 6 → 8 → 2
    W1, b1 = initialize_weights(3, 6)
    W2, b2 = initialize_weights(6, 8)
    W3, b3 = initialize_weights(8, 2)
    
    network = Chain(
        Dense(W1, b1, tanh),
        Dense(W2, b2, tanh),
        Dense(W3, b3, identity)
    )
    return network
end

function analyze_distribution(network, n_samples=10000)
    # Gera dados de entrada aleatórios
    inputs = randn(3, n_samples)
    
    # Passa pelas camadas e coleta ativações
    l1 = network[1](inputs)
    l2 = network[2](l1)
    outputs = network[3](l2)
    
    # Calcula estatísticas
    stats = Dict()
    stats["input_mean"] = mean(inputs)
    stats["input_var"] = var(inputs)
    
    stats["l1_mean"] = mean(l1)
    stats["l1_var"] = var(l1)
    
    stats["l2_mean"] = mean(l2)
    stats["l2_var"] = var(l2)
    
    stats["output_mean"] = mean(outputs)
    stats["output_var"] = var(outputs)
    
    return stats, (inputs, l1, l2, outputs)
end

function plot_distributions(data)
    inputs, l1, l2, outputs = data
    
    p1 = histogram(vec(inputs), title="Input Distribution", label="Input")
    p2 = histogram(vec(l1), title="Layer 1 (6 neurons)", label="Layer 1")
    p3 = histogram(vec(l2), title="Layer 2 (8 neurons)", label="Layer 2")
    p4 = histogram(vec(outputs), title="Output (2 neurons)", label="Output")
    
    plot(p1, p2, p3, p4, layout=(2, 2), size=(800, 600))
end

function run_experiment()
    # Cria a rede
    network = create_network()
    
    # Analisa a distribuição
    stats, data = analyze_distribution(network)
    
    # Exibe estatísticas
    println("Estatísticas das Distribuições:")
    for (k, v) in stats
        println("$k: ", round(v, digits=4))
    end
    
    # Plota as distribuições
    plot_distributions(data)
end

# Executa o experimento
run_experiment()