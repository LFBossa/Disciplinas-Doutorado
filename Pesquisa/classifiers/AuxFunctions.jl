module AuxFunctions

using DataFrames, ARFFFiles, Glob

export carregar_arff_pasta, args2dict, load_datasets_from_txt

function load_datasets_from_txt(filename::String)
    datasets = []
    if isfile(filename)
        open(filename, "r") do f
            for line in eachline(f)
                line = strip(line)
                if !isempty(line) && !startswith(line, "#")
                    push!(datasets, line)
                end
            end
        end
    end
    return datasets
end

function carregar_arff_pasta(caminho_pasta::String)::DataFrame
    """
    Carrega todos os arquivos .arff de uma pasta em um único DataFrame.
    
    Args:
        caminho_pasta: Caminho da pasta contendo arquivos .arff
    
    Returns:
        DataFrame com dados consolidados de todos os arquivos
    """
    
    # Listar todos os arquivos .arff na pasta
    arquivos_arff = glob("*.arff", caminho_pasta)
    
    if isempty(arquivos_arff)
        @warn "Nenhum arquivo .arff encontrado em $caminho_pasta"
        return DataFrame()
    end
    
    # Carregar primeiro arquivo
    df_consolidado = ARFFFiles.load(DataFrame, arquivos_arff[1])
    
    # Carregar e concatenar arquivos restantes
    for arquivo in arquivos_arff[2:end]
        df_temp = ARFFFiles.load(DataFrame, arquivo)
        df_consolidado = vcat(df_consolidado, df_temp; cols=:union)
    end
    
    return df_consolidado
end


function args2dict(args :: Vector{String}) :: Dict{String, Union{String, Nothing}}
    """Converte um vetor de strings no formato "chave=valor" em um dicionário."""
    dict = Dict()
    for string in args
        if occursin("=", string)
            key, value = split(string, "=", limit=2)
            dict[key] = value
        else
            dict[string] = nothing
        end
    end
    return dict
end

end # module AuxFunctions
