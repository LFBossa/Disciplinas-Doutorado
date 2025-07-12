using Plots
using Flux, ProgressMeter, Statistics
using CUDA  # optional
using BSON  # for saving/loading models
device = gpu_device()  # function to move data and model to the GPU


BSON.@load "model_n1.bson" cpu_model  # load the model from BSON file
model1 = cpu_model

BSON.@load "model_n2.bson" cpu_model  # load the model from BSON file
model2 = cpu_model

BSON.@load "data_n2.bson" x_data1 y_data1  # loa
my_function(x) = 2x - x^3 + cos(x^3)  # the function we want to fit


# Sort x and y data for plotting 
 

x_range = collect(Float32, -3:0.1f0:3)  # range of x values

y_range1 = model1(x_range')  # evaluate model on the range of x values
y_range2 = model2(x_range')  # evaluate model on the range of x values
y_range1 = y_range1[1, :]  # extract the first row (output)
y_range2 = y_range2[1, :]  # extract the first row (output 

diff = y_range2 - y_range1  # calculate the difference between the two models

plot(my_function, -3, 3, label="Função verdadeira", color=:purple, linewidth=2)
plot!(x_range, y_range1, label="Modelo 1", color=:blue, linewidth=2)
plot!(x_range, y_range2, label="Modelo 2", color=:red, linewidth=2) 