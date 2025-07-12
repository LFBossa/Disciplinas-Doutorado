using Plots
using Flux, Statistics, ProgressMeter
using CUDA  # optional
device = gpu_device()  # function to move data and model to the GPU


function para_aproximar(x, y)
    exp(-sin(x^2 + y^2)) / (x^2 + y^2 + 1) # a function to approximate
end

xs = collect(-3:0.1:3)  # x values
ys = collect(-3:0.1:3)  # y values
X = [x for x = xs for _ in ys]  # x coordinates for the surface
Y = [y for _ in xs for y = ys]  # y coordinates for the
Z = [para_aproximar(x, y) for x in X, y in Y]  # z values for the surface

surface(X, Y, para_aproximar.(X, Y), xlabel="X", ylabel="Y", zlabel="Z", title="Surface Plot of para_aproximar")#, color=:viridis)

D = 300
points = randn(Float32, 2, D)  # generate random points in 2D
data = [([x, y], para_aproximar(x, y)) for (x, y) in eachcol(points)]  # create pairs of (x, y) and z values


# Para o Flux, normalmente usamos tuplas (entrada, saída) onde entrada e saída são vetores coluna


model = Chain(
    Dense(2 => D, tanh),      # hidden layer with D neurons
    Dense(D => 4*D, tanh),    # another hidden layer with 4D neurons
    Dense(4*D => D, tanh),    # another hidden layer with D neurons
    Dense(D => 1)) |> device  # output layer for single value

 
optimizer = Flux.setup(Adam(0.01), model)

meanloss(model, x, yhat) = Flux.mse(model(x), yhat)  # mean squared error loss function

losses = []  # to store losses during training 

for lr in [2.0^-k for k=9:12]  # learning rates to try
    println("Training with learning rate: $lr for 100 epochs")
    Flux.adjust!(optimizer, lr)  # set learning rate
    @showprogress for epoch in 1:100
        for (x, y) in data
            x_gpu = x |> device  # move input to GPU if available
            y_gpu = y |> device  # move target to GPU if available
            loss, grads = Flux.withgradient(model) do m
                meanloss(m, x_gpu, y_gpu)  # calculate loss and gradients
            end
            Flux.update!(optimizer, model, grads[1])  # update model parameters
            push!(losses, loss)  # store the loss
        end
    end
    # Calculate and store the loss for this epoch 
end


# Plot the fitted surface
Z_fit = model(hcat(X, Y)' |> device) |> cpu # get the model's predictions for the grid
Z_fit = reshape(Z_fit, size(Z,2))  # reshape to match the grid dimensions
surface(X, Y, Z_fit, label="Fitted Surface", alpha=0.5)
 
plot(losses, xaxis=(:log10,"epochs"),yaxis=(:log10,"loss"), title="Training Loss", xlabel="Epochs", ylabel="Loss", label="Loss", legend=:topright)
