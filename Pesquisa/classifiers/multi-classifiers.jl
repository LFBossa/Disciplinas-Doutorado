using MLJ 
using Random


# Carregando os modelos 

SVC = @load SVC pkg=LIBSVM
KNNClassifier = @load KNNClassifier pkg=NearestNeighborModels
NeuralNetworkClassifier = @load NeuralNetworkClassifier pkg=MLJFlux
GaussianNBClassifier = @load GaussianNBClassifier pkg=MLJScikitLearnInterface
DecisionTreeClassifier = @load DecisionTreeClassifier pkg=DecisionTree

X, y = @load_iris # a table and a vector from the iris dataset


## TODO mudar a abordagem para dataframe

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

train_index = perclass_splits(y, 0.84)

test_index = setdiff(1:length(y), train_index)

y_train = y[train_index]

y_test = y[test_index]

X_train_dict = Dict{Symbol, Vector{Float64}}()
X_test_dict = Dict{Symbol, Vector{Float64}}()

for key in keys(X)
    X_train_dict[key] = X[key][train_index]
    X_test_dict[key] = X[key][test_index]
end

X_train = NamedTuple(X_train_dict)
X_test = NamedTuple(X_test_dict)


## Modelo 1: SVC

model1 = SVC()
mach1 = machine(model1, X_train, y_train)
fit!(mach1)

predict1 = predict(mach1, X_test)

confusion_matrix(y_test, predict1)

## Modelo 2: KNN

model2 = KNNClassifier(K=3) 
mach2 = machine(model2, X_train, y_train)
fit!(mach2)
predict2 = predict_mode(mach2, X_test)

confusion_matrix(y_test, predict2)

## Modelo 3: Neural Network

model3 = NeuralNetworkClassifier(builder = MLJFlux.MLP(; hidden = (5,)), epochs = 100)
mach3 = machine(model3, X_train, y_train)
fit!(mach3)
predict3 = predict_mode(mach3, X_test)

confusion_matrix(y_test, predict3)

## Modelo 4: Gaussian Naive Bayes

model4 = GaussianNBClassifier()
mach4 = machine(model4, X_train, y_train)
fit!(mach4)
predict4 = predict_mode(mach4, X_test)

confusion_matrix(y_test, predict4)


## Modelo 5: Decision Tree

model5 = DecisionTreeClassifier(max_depth = 3)
mach5 = machine(model5, X_train, y_train)
fit!(mach5)
predict5 = predict_mode(mach5, X_test)

confusion_matrix(y_test, predict5)

for label ∈ unique(predict5)
    println("Label: $label")
    println("Count: ", count(==(label), predict5))
end

X_test_dict

using DataFrames

X_test_df = DataFrame(X_test)

predict_mode(mach2, X_test_df)  # KNN mean probabilities
