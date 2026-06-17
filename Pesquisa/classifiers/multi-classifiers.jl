##% Celula
using MLJ 
using Random
using DataFrames
include("AuxFunctions.jl")
include("ensemble.jl")
using .AuxFunctions: carregar_arff_pasta

##% 
# Carregando os modelos 

SVC = @load SVC pkg=LIBSVM
KNNClassifier = @load KNNClassifier pkg=NearestNeighborModels
NeuralNetworkClassifier = @load NeuralNetworkClassifier pkg=MLJFlux
GaussianNBClassifier = @load GaussianNBClassifier pkg=MLJScikitLearnInterface
DecisionTreeClassifier = @load DecisionTreeClassifier pkg=DecisionTree
 
# Exemplo de uso: winequality-red
df = carregar_arff_pasta("datasets/ordinal-regression/winequality-red/weka/")
ultima_coluna = Symbol(names(df)[end])
X = df[:, Not(ultima_coluna)]
y = df[:, ultima_coluna];

df = carregar_arff_pasta("datasets/ordinal-regression/eucalyptus/weka/")
X = df[:, Not(:Utility)]
y = df[:, :Utility];

levels_dict = Dict(i => c for (i, c) in enumerate(levels(y)))
idx_to_class = x -> levels_dict[x]


y_num = [levels_dict[level] for level in y]

function perclass_splits(y, percent)
    uniq_class = unique(y)
    keep_index = []
    for class in uniq_class
        class_index = findall(y .== class)
        row_index = randsubseq(class_index, percent)
        push!(keep_index, row_index...)
    end
    return keep_index
end

# Spliting train and test data



Random.seed!(26)

train_index = perclass_splits(y, 0.7)

test_index = setdiff(1:length(y), train_index)

 

train_index, test_index = partition(eachindex(y), 0.8, shuffle=true, rng=Random.MersenneTwister(26))

# y_train = y[train_index]

# y_test = y[test_index]

# X_train = Dataset[train_index, Not(:y)]

# X_test = Dataset[test_index, Not(:y)]


## Modelo 1: SVC

model1 = SVC()
mach1 = machine(model1, X, y)
fit!(mach1, rows=train_index)

predict1 = predict(mach1, rows=test_index)

confusion_matrix(predict1, y[test_index])

## Modelo 2: KNN

model2 = OneHotEncoder() |> KNNClassifier(K=3) 
mach2 = machine(model2, X, y)
fit!(mach2, rows=train_index)
predict2 = predict_mode(mach2, rows=test_index)

confusion_matrix(predict2, y[test_index])
misclassification_rate(predict2, y[test_index])

## Modelo 3: Neural Network

model3 = OneHotEncoder() |> NeuralNetworkClassifier(builder = MLJFlux.MLP(; hidden = (10,15,)), epochs = 100)
mach3 = machine(model3, X, y)
fit!(mach3, rows=train_index)
predict3 = predict_mode(mach3, rows=test_index)

test_index
confusion_matrix(predict3,y[test_index])

## Modelo 4: Gaussian Naive Bayes

model4 = OneHotEncoder() |> GaussianNBClassifier()
mach4 = machine(model4, X, y)
fit!(mach4, rows=train_index)
predict4 = predict_mode(mach4, rows=test_index)

confusion_matrix(predict4, y[test_index])
misclassification_rate(predict4, y[test_index])

## Modelo 5: Decision Tree

model5 = OneHotEncoder() |> DecisionTreeClassifier(max_depth = -1)
mach5 = machine(model5, X, y)
fit!(mach5, rows=train_index)
predict5 = predict_mode(mach5, rows=test_index)

confusion_matrix(predict5, y[test_index])

misclassification_rate(predict5, y[test_index])
 
## constroi Y e C para o ensemble
function constructYC(classifiers, X_test, y_test)
    n = length(y_test)
    T = length(classifiers)
    unique_classes = levels(y_test)
    K = length(unique_classes)
    class_dict = Dict(c => i for (i, c) in enumerate(unique_classes))
    class_to_index = x -> class_dict[x]
    Y = zeros(K, T, n)
    C = class_to_index.(y_test)
    for t in 1:T
        pred = nothing
        try
            pred = predict_mode(classifiers[t], X_test)
        catch
            pred = predict(classifiers[t], X_test)
        end
        pred_index = class_to_index.(pred)
        for i in 1:n
            k = pred_index[i]
            Y[k, t, i] = 1
        end
    end 
    return Y, C
end

classifiers = [mach2, mach4, mach5]

Y, C = constructYC(classifiers, X[test_index,:], y[test_index])
 

ω2,ξ2,fun2 = Ensemble2(Y,C,0.05)

ω1,ξ1,fun1 = Ensemble1(Y,C,0.05)