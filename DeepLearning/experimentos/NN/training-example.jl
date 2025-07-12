using Flux, Statistics, ProgressMeter
using CUDA  # optional
using Plots  # to draw the above figure
using BSON
using Printf  # for @sprintf
device = gpu_device()  # function to move data and model to the GPU

function get_first_arg_as_string()
    if length(ARGS) < 1
        error("Nenhum argumento fornecido.")
    end
    return @sprintf("%02d", parse(Int, ARGS[1]))
end

caracteristica(col) = begin
    # x = col[1], y = col[2]
    if abs(col[1]) > 1 || abs(col[2]) > 1
        return false  # if either coordinate is outside [-1,1]
    end
    if abs(col[2]) > 0.8
        return true  # if y is close to 1, then it is a true positive
    end
    raio = col[1]^2 + col[2]^2  # distance from origin
    if raio < 0.5 && raio > 0.25
        return true  # if inside the circle of radius 0.5, then it is a true positive
    end
    false # otherwise, it is a false positive
end  # function to classify each column

N_MODELO = get_first_arg_as_string()  # read the model name from the command line

println("Using model: ", N_MODELO)

# Generate some data for the XOR problem: vectors of length 2, as columns of a matrix:
noisy = randn(Float32, 2, 1000)                                    # 2×1000 Matrix{Float32}
truth = [caracteristica(col) for col in eachcol(noisy)]   # 1000-element Vector{Bool}

BSON.@save "data_$(N_MODELO).bson" noisy truth  # save the data to a BSON file

# Define our model, a multi-layer perceptron with one hidden layer of size 3:
model = Chain(
    Dense(2 => 100, relu),      # activation function inside layer
    Dense(100 => 300, relu),    # another hidden layer
    Dense(300 => 2)) |> device  # move model to GPU, if one is available

# The model encapsulates parameters, randomly initialised. Its initial output is:
out1 = model(noisy |> device)    # 2×1000 Matrix{Float32}, or CuArray{Float32}
probs1 = softmax(out1) |> cpu    # normalise to get probabilities (and move off GPU)

# To train the model, we use batches of 64 samples, and one-hot encoding:
target = Flux.onehotbatch(truth, [true, false])                   # 2×1000 OneHotMatrix
loader = Flux.DataLoader((noisy, target), batchsize=32, shuffle=true);

opt_state = Flux.setup(Flux.Adam(0.005), model)  # will store optimiser momentum, etc.

# Training loop, using the whole data set 1000 times:
losses = []
@showprogress for epoch in 1:1_000
    for xy_cpu in loader
        # Unpack batch of data, and move to GPU:
        x, y = xy_cpu |> device
        loss, grads = Flux.withgradient(model) do m
            # Evaluate model and loss inside gradient context:
            y_hat = m(x)
            Flux.logitcrossentropy(y_hat, y)
        end
        Flux.update!(opt_state, model, grads[1])
        push!(losses, loss)  # logging, outside gradient context
    end
end

opt_state # parameters, momenta and output have all changed

out2 = model(noisy |> device)         # first row is prob. of true, second row p(false)
probs2 = softmax(out2) |> cpu         # normalise to get probabilities
mean((probs2[1, :] .> 0.5) .== truth)  # accuracy 94% so far!

p_true = scatter(noisy[1, :], noisy[2, :], markersize=3, zcolor=truth, title="True classification", legend=false)
p_raw = scatter(noisy[1, :], noisy[2, :], markersize=3, zcolor=probs1[1, :], title="Untrained network", label="", clims=(0, 1))
p_done = scatter(noisy[1, :], noisy[2, :], markersize=3, zcolor=probs2[1, :], title="Trained network", legend=false)

BSON.@save "modelo_$(N_MODELO).bson" model

savefig(plot(p_true, p_raw, p_done, layout=(1, 3), size=(2000, 660)), "trained_$(N_MODELO).png")

plt_loss = plot(losses; xaxis=(:log10, "iteration"),
    yaxis=(:log10, "loss"), label="per batch")
n = length(loader)
plot!(plt_loss, n:n:length(losses), mean.(Iterators.partition(losses, n)),
    label="epoch mean", dpi=200)
savefig(plt_loss, "loss_$(N_MODELO).png")

# plot(p_true, p_raw, p_done, layout=(1,3), size=(1000,330))

# plot(losses; xaxis=(:log10, "iteration"),
#     yaxis="loss", label="per batch")
# n = length(loader)
# plot!(n:n:length(losses), mean.(Iterators.partition(losses, n)),
#     label="epoch mean", dpi=200)