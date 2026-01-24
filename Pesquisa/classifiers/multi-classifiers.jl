using MLJ 
using Random
using DataFrames

# Carregando os modelos 

SVC = @load SVC pkg=LIBSVM
KNNClassifier = @load KNNClassifier pkg=NearestNeighborModels
NeuralNetworkClassifier = @load NeuralNetworkClassifier pkg=MLJFlux
GaussianNBClassifier = @load GaussianNBClassifier pkg=MLJScikitLearnInterface
DecisionTreeClassifier = @load DecisionTreeClassifier pkg=DecisionTree

X, y = @load_iris # a table and a vector from the iris dataset

Dataset = hcat(DataFrame(X), DataFrame(y = y))

# define function to split data (source: Huda Nassar)

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

X = df[:, Not(:Quality)]
y = df[:, :Quality]
train_index, test_index = partition(eachindex(y), 0.7, shuffle=true, rng=Random.MersenneTwister(26))

# y_train = y[train_index]

# y_test = y[test_index]

# X_train = Dataset[train_index, Not(:y)]

# X_test = Dataset[test_index, Not(:y)]


## Modelo 1: SVC

model1 = SVC()
mach1 = machine(model1, X, y)
fit!(mach1, rows=train_index)

predict1 = predict(mach1, rows=test_index)

confusion_matrix(y[test_index], predict1)

## Modelo 2: KNN

model2 = KNNClassifier(K=6) 
mach2 = machine(model2, X, y)
fit!(mach2, rows=train_index)
predict2 = predict_mode(mach2, rows=test_index)

confusion_matrix(y[test_index], predict2)


## Modelo 3: Neural Network

model3 = NeuralNetworkClassifier(builder = MLJFlux.MLP(; hidden = (10,10,15,)), epochs = 100)
mach3 = machine(model3, X, y)
fit!(mach3, rows=train_index)
predict3 = predict_mode(mach3, rows=test_index)

test_index
confusion_matrix(y[test_index], predict3)

## Modelo 4: Gaussian Naive Bayes

model4 = GaussianNBClassifier()
mach4 = machine(model4, X, y)
fit!(mach4, rows=train_index)
predict4 = predict_mode(mach4, rows=test_index)

confusion_matrix(y[test_index], predict4)
misclassification_rate(y[test_index], predict4)

## Modelo 5: Decision Tree

model5 = DecisionTreeClassifier(max_depth = 6)
mach5 = machine(model5, X, y)
fit!(mach5, rows=train_index)
predict5 = predict_mode(mach5, rows=test_index)

confusion_matrix(y[test_index], predict5)
 
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

classifiers = [mach1, mach2, mach3, mach4, mach5]

Y, C = constructYC(classifiers, X[test_index,:], y[test_index])

sum(Y; dims=3) # verifica se cada classificador atribuiu uma classe a cada ponto


ω2,ξ2,fun2 = Ensemble2(Y,C,0.1)
