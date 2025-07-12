using Plots
using Flux, ProgressMeter, Statistics
using CUDA  # optional
using BSON  # for saving/loading models
device = gpu_device()  # function to move data and model to the GPU


BSON.@load "model_n1.bson" cpu_model  # load the model from BSON file
BSON.@load "data_n1.bson" x_data1 y_data1  # load the data from BSON file
D = 50 
my_function(x) = 2x - x^3 + cos(x^3)# the function we want to fit

model = cpu_model |> device  # move model to GPU, if one is available


optimizer = Flux.setup(Adam(0.0001, (0.9,0.8)), model)
data_loader = Flux.DataLoader((x_data1, y_data1), batchsize=25, shuffle=true)  # create a data loader


losses = [] 
convergiu = false
@showprogress for epoch in 1:100_000
    for (x,y) in data_loader
        x_gpu = x' |> device  # move input to GPU if available
        y_gpu = y' |> device  # move target to GPU if available 
        loss, grads = Flux.withgradient(model) do m
            yhat = m(x_gpu)  # evaluate model on the batch of data
            Flux.mse(y_gpu, yhat)  # calculate loss and gradients
        end
        Flux.update!(optimizer, model, grads[1])  # update model parameters
        push!(losses, loss)  # log the loss
        if epoch % 1000 == 0
            @info "Epoch $epoch: Loss = $loss"
        end
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


cpu_model = model |> cpu  # move model back to CPU for saving
@info "Model moved to CPU for saving."
BSON.@save "model_n1.bson" cpu_model  # save the trained model
@info "Model and data saved successfully."
 
