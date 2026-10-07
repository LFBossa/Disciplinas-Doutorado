include("batch-classifiers.jl")
include("AuxFunctions.jl")


PARAMS = args2dict(ARGS)
 
version = "v5" 

DATASETS = load_datasets_from_txt("datasets.txt")

PTCONFIGS = Dict(
    1 => 3:10,
    2 => 3:5,
    3 => 3:3,
)

for P in 1:3
    for T in PTCONFIGS[P]
        RND_SEED = parse(Int, PARAMS["SEED"])
        @info "Running batch classifiers for version=$version, P=$P, rnd_seed=$RND_SEED, T=$T"
        batchclassifiers(DATASETS, version, RND_SEED, P, T)
    end
end 
