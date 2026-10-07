include("reading-results.jl")
include("AuxFunctions.jl")
using Glob
#using .AuxFunctions: args2dict, load_datasets_from_txt

DATASETS = load_datasets_from_txt("datasets.txt")


PTCONFIGS = Dict(
    1 => [3, 6, 10],
    2 => [3, 4, 5],
    3 => [3],
)

PTCONFIGS = Dict(
    1 => [7],
    2 => [5],
    3 => [3],
)

DATASETS_PATH = load_datasets_from_txt("datasets.txt")
DATASETS = [split(dataset_path, "/")[end] for dataset_path in DATASETS_PATH]

for dataset in DATASETS
    print("$dataset\t")
    for v in ["v3", "v4", "v5"]
        i = 0
        for P in 1:3
            for T in PTCONFIGS[P]
                print("&")
                versions = glob("results/json/$dataset:$v:P$P:*:T$T.json")
                DICT_LOGS = [JSON.parsefile(log_path) for log_path in versions]
                statitics = [dict_log["tstatistic"] for dict_log in DICT_LOGS]
                wins = sum([x > 0 ? 1 : 0 for x in statitics])
                ties = sum([x == 0 ? 1 : 0 for x in statitics])
                losses = sum([x < 0 ? 1 : 0 for x in statitics])
                if wins > losses
                    print("\\textcolor{blue}{$wins:$ties:$losses}")
                elseif losses > wins
                    print("\\textcolor{gray}{$wins:$ties:$losses}")
                else
                    print("$wins:$ties:$losses")
                end
                i += 1
            end
        end
        print("\t")
        #println(i)
    end
    println("\\\\")
end