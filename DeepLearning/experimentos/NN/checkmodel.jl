using Plots
using Flux, ProgressMeter, Statistics
using CUDA  # optional
using BSON  # for saving/loading models
device = gpu_device()  # function to move data and model to the GPU


BSON.@load "model_n1.bson" cpu_model  # load the model from BSON file


x_range = collect(Float32, -3:0.1f0:3)  # range of x values

y_range = cpu_model(x_range')  # evaluate model on the range of x values
y_range = y_range[1, :]  # extract the first row (output)

plot(x_range, y_range, label="Model Prediction", color=:blue, linewidth=2, title="Model Predictions vs. True Function", xlabel="x", ylabel="y")
my_function(x) = 2x - x^3 + cos(x^3)  # the function we want to fit
plot!(my_function, -3, 3, label="True Function", color=:purple, linewidth=2)
