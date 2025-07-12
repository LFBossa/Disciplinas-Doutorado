using Plots
using Flux, ProgressMeter, Statistics
using CUDA  # optional
using BSON  # for saving/loading models
device = gpu_device()  # function to move data and model to the GPU


n1, n2, n3 = 10, 100, 10  # number of neurons in each layer

model = Chain(
    Dense(1,n1,tanh), 
    Dense(n1,n2,tanh), 
    Dense(n2,n3,tanh),
    Dense(n3,1)) |> device  # move model to GPU, if one is available

my_function(x) = 2x - x^3 + cos(x^3)# the function we want to fit

x_range = collect(Float32, -3:0.1f0:3)  # range of x values

D = 100

x_data = randn(Float32, D)  # generate D random x values

data = [([x], my_function(x)) for x in x_data]  # training points (x, y)

BSON.@save "data.bson" data  # save the data for later use

optimizer = Flux.setup(Adam(0.001), model)
 


function scattermodel!(x)
    x_gpu = x' |> device  # move input to GPU if available
    model_cpu = model(x_gpu) |> cpu  # evaluate model and move output back to CPU
    model_cpu = model_cpu[1, :]  # extract the first row (output)
    scatter!(x, model_cpu, label="", markersize=5, color=:blue)
end 

data_loader = Flux.DataLoader((x_data, my_function.(x_data)), batchsize=30, shuffle=true)  # create a data loader
 

losses = [] 
convergiu = false
@showprogress for epoch in 1:1_000_000
    for (x,y) in data_loader
        x_gpu = x' |> device  # move input to GPU if available
        y_gpu = y' |> device  # move target to GPU if available 
        loss, grads = Flux.withgradient(model) do m
            yhat = m(x_gpu)  # evaluate model on the batch of data
            Flux.mse(y_gpu, yhat)  # calculate loss and gradients
        end
        Flux.update!(optimizer, model, grads[1])  # update model parameters
        push!(losses, loss)  # log the loss
        if loss < 1e-10
            @info "Early stopping at epoch $epoch with loss $loss"
            global convergiu = true  # stop if loss is sufficiently low
            break
        end
    end
    if convergiu
        @info "Convergence achieved at epoch $epoch"
        break  # exit the loop if convergence is achieved
    end
end  
 

 
figure = plot(my_function, -3,3, label="truth", dpi=600)
#scattermodel!(x_data)  # plot the model's predictions over the range of x values
scattermodel!(x_data)  # plot the model's predictions over the range of x values
savefig(figure, "trained_model.png") 

figure2 = plot(losses; xaxis=(:log10, "iteration"),
    yaxis=(:log10,"loss"), label="per batch")

plot!(collect(1:D:length(losses)), 
    mean.(Iterators.partition(losses, D)),
    label="epoch mean", dpi=200)
 
savefig(figure2, "losses.png")

cpu_model = model |> cpu  # move model back to CPU for saving
@info "Model moved to CPU for saving."
BSON.@save "model.bson" cpu_model  # save the trained model
@info "Model and data saved successfully."