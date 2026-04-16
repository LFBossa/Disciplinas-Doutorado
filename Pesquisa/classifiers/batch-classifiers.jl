using MLJ
using Random
using DataFrames
include("AuxFunctions.jl")
include("ensemble.jl")
using .AuxFunctions: carregar_arff_pasta
using JSON
# Carregando os modelos 

DecisionTreeClassifier = @load DecisionTreeClassifier pkg = DecisionTree

TRAIN_PERCENTAGE = 1
TRAIN_P = TRAIN_PERCENTAGE / 100
T = 15
version = "v3"
RND_SEED = 651


# Ler arquivo datasets.txt e preencher array DATASETS
DATASETS = []
if isfile("datasets.txt")
    open("datasets.txt", "r") do f
        for line in eachline(f)
            line = strip(line)
            if !isempty(line) && !startswith(line, "#")
                push!(DATASETS, line)
            end
        end
    end
else
    # Se datasets.txt não existir, usar ARGS como fallback
    DATASETS = [ARGS[1]]
end


for data_path in DATASETS

    dataset = split(data_path, "/")[end]
    @info "Carregando dataset: $dataset"
    df = carregar_arff_pasta("$data_path/weka/")
    ultima_coluna = Symbol(names(df)[end])
    X = df[:, Not(ultima_coluna)]
    y = df[:, ultima_coluna] 

    classes = levels(y)
    @info classes

    @info names(df)

    K = length(classes)

    Random.seed!(26)


    y_dict = Dict(i => c for (i, c) in enumerate(classes))
    y_int = Dict(c => i for (i, c) in enumerate(classes))
    idx_to_class = x -> y_dict[x]
    class_to_idx = x -> y_int[x]






    MACHINES = Array{Any}(undef, T)
    CONFUSIONS = Array{Any}(undef, T)
    MISCLASS_RATES = Array{Number}(undef, T)
    MAE = Array{Number}(undef, T)
    train_indexes..., test_index = partition(eachindex(y),
        [TRAIN_P for _ in 1:T]...,
         shuffle=true, rng=RND_SEED)
    # Vamos treinar o ensemble usando apenas os dados de treino, e depois avaliar no teste
    ensemble_index = vcat(train_indexes...)
    for i in 1:T
        @info "Treinando modelo $i..."
        # Dividimos o conjunto de dados em treino 5% por modelo, 25% para ensemble e 50% para teste
        model = OneHotEncoder() |> DecisionTreeClassifier(max_depth=K)
        mach = machine(model, X, y)
        MLJ.fit!(mach, rows=train_indexes[i])
        MACHINES[i] = mach
        @info "Avaliando modelo $i..."
        predict = predict_mode(mach, rows=test_index)
        cm = confusion_matrix(predict, y[test_index])
        misclass = misclassification_rate(predict, y[test_index])
        MAE[i] = mean(abs.(class_to_idx.(predict) .- class_to_idx.(y[test_index])))
        CONFUSIONS[i] = cm
        MISCLASS_RATES[i] = misclass
    end


    ## constroi Y e C para o ensemble



    Y, C = constructYC(MACHINES, X[ensemble_index, :], y[ensemble_index])

    TOLERANCE = 0.1

    ω2, ξ2, fun2, model2 = Ensemble2(Y, C, TOLERANCE)
    pred_ensemble2 = prediction_from_ensemble_class(Y, ω2, idx_to_class)
    cm_ensemble2 = confusion_matrix(pred_ensemble2, y[ensemble_index])
    misclass_ensemble2 = misclassification_rate(pred_ensemble2, y[ensemble_index])

    ω1, ξ1, fun1, model1 = Ensemble1(Y, C, TOLERANCE)
    pred_ensemble1 = prediction_from_ensemble_class(Y, ω1, idx_to_class)
    cm_ensemble1 = confusion_matrix(pred_ensemble1, y[ensemble_index])
    misclass_ensemble1 = misclassification_rate(pred_ensemble1, y[ensemble_index])

    Y_validation, C_validation = constructYC(MACHINES, X[test_index, :], y[test_index])

    y_num_val = class_to_idx.(y[test_index])
    pred1 = prediction_from_ensemble_int(Y_validation, ω1)
    pred2 = prediction_from_ensemble_int(Y_validation, ω2) 
 


    AE1 = abs.(pred1 .- y_num_val)
    MAE1 = mean(AE1)
    SD1 = std(AE1)

    AE2 = abs.(pred2 .- y_num_val)
    MAE2 = mean(AE2)
    SD2 = std(AE2)


    dicionario_log = Dict(
        "data_size" => size(df),
        "num_classes" => K,
        "dataset" => data_path,
        "individual_models" => [
            Dict(
                "model_index" => i,
                "seed_value" => 26,
                "confusion_matrix" => CONFUSIONS[i],
                "misclassification_rate" => MISCLASS_RATES[i],
                "mae" => MAE[i]
            ) for i in 1:length(MACHINES)
        ],
        "ensemble_model_1" => Dict(
            "confusion_matrix" => cm_ensemble1,
            "misclassification_rate" => misclass_ensemble1,
            "weights" => ω1,
            "func_val" => fun1,
            "mae" => MAE1,
            "std" => SD1,
            "time" => MOI.get(model1, MOI.SolveTimeSec())
        ),
        "ensemble_model_2" => Dict(
            "confusion_matrix" => cm_ensemble2,
            "misclassification_rate" => misclass_ensemble2,
            "weights" => ω2,
            "func_val" => fun2,
            "mae" => MAE2,
            "std" => SD2,
            "time" => MOI.get(model2, MOI.SolveTimeSec())
        )
    )
 
    # Salvando resultados em arquivo de log 
    log_file = "results/json/$dataset:$version:P$TRAIN_PERCENTAGE:S$RND_SEED:T$T.json"
    open(log_file, "w") do io
        JSON.print(io, dicionario_log, 2)
    end
    @info "Resultados salvos em: $log_file"
end