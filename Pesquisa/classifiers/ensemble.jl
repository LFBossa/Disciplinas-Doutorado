## Ensemble.jl 
using JuMP, Ipopt
using LinearAlgebra: dot
using MLJ 


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

function Ensemble2(Y,C,ϵ)
    #Y[k,t,i] = 1 if tree t classify x^i as k, 0 otherwise
    #C in R^n: classification of each point
    #ϵ: parameter
    K,T,n = size(Y)
    model = JuMP.Model(Ipopt.Optimizer)
    #set_silent(model)
    @variable(model, ω[t=1:T]≥0)
    @variable(model, ξ[g=1:n]≥0)
    for i ∈ 1:n 
            k_i = C[i] #classe of point
            k_dif = deleteat!(collect(1:K),k_i) #all other classes  
            @constraint(model, [k ∈ k_dif], dot(ω,Y[k_i,:,i]) ≥ dot(ω,Y[k,:,i]) - ξ[i]/abs(k_i-k) + ϵ) 
    end
    @constraint(model, sum(ω) == 1) 
    @objective(model, Min, sum(ξ))
    print(model)

    optimize!(model)
  
    ω, ξ, fun = value.(ω), value.(ξ), objective_value(model)

    return ω,ξ,fun, model 
end

## Ensemble1: Basic version
function Ensemble1(Y,C,ϵ)
    #Y[k,t,i] = 1 if tree t classify x^i as k, 0 otherwise
    #C in R^n: classification of each point
    #ϵ: parameter
    K,T,n = size(Y)
    model = JuMP.Model(Ipopt.Optimizer)
    #set_silent(model)
    @variable(model, ω[t=1:T]≥0)
    @variable(model, ξ[g=1:n]≥0)
    for i ∈ 1:n 
            k_i = C[i] #classe of point
            k_dif = deleteat!(collect(1:K),k_i) #all other classes  
            @constraint(model, [k ∈ k_dif], dot(ω,Y[k_i,:,i]) ≥ dot(ω,Y[k,:,i]) - ξ[i] + ϵ)
    end
    @constraint(model, sum(ω) == 1) 
    @objective(model, Min, sum(ξ))
    set_attribute(model, "print_level", 5)
    print(model)
    optimize!(model)
  
    ω, ξ,fun = value.(ω), value.(ξ), objective_value(model)

    return ω,ξ,fun, model
end

function prediction_from_ensemble_class(Y, ω, int_to_class_fn)
    K, T, n = size(Y)
    new_predictions = zeros(Int, n)
    for i in 1:n
        votes = Y[:, :, i] * ω
        predicted_class_index = argmax(votes)
        new_predictions[i] = predicted_class_index
    end
    return int_to_class_fn.(new_predictions)
end

function prediction_from_ensemble_int(Y, ω)
    K, T, n = size(Y)
    new_predictions = zeros(Int, n)
    for i in 1:n
        votes = Y[:, :, i] * ω
        predicted_class_index = argmax(votes)
        new_predictions[i] = predicted_class_index
    end
    return new_predictions
end

function ordered_loss(prediction1, prediction2, true_labels) 
    n = length(true_labels)
    loss1 = prediction1 - true_labels
    loss2 = prediction2 - true_labels 

    return loss1, loss2
end