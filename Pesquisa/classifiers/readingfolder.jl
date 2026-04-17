using Glob
include("reading-results.jl")
include("AuxFunctions.jl")

#using .AuxFunctions: args2dict, load_datasets_from_txt

versions = glob("results/json/automobile*.json")


CONFIGS_ARRAY = []
for version in versions
    split_path = split(version, ":")
    version = split_path[2]
    P = parse(Int, split_path[3][2:end])
    SEED = parse(Int, split_path[4][2:end])
    split_T = split(split_path[5], ".")[1]
    T = parse(Int, split_T[2:end])
    push!(CONFIGS_ARRAY, (version=version, P=P, SEED=SEED, T=T))
end

for config in CONFIGS_ARRAY 
    @info "Reading results for version=$config.version, P=$config.P, rnd_seed=$config.SEED, T=$config.T"
    readresults(DATASETS,  config.version, config.SEED, config.P, config.T)
end

