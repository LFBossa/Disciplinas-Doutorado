using JSON
using DataFrames
using PrettyTables
using XLSX
using Statistics

include("AuxFunctions.jl")
using .AuxFunctions: args2dict, load_datasets_from_txt




DATASETS = load_datasets_from_txt("datasets.txt")

DATASETS = [split(ds, "/")[end] for ds in DATASETS]



function readresults(DATASETS, version, rnd_seed, P, T)

    DATALOGS = ["results/json/$dataset:$version:P$P:S$rnd_seed:T$T.json" for dataset in DATASETS]

    DICT_LOGS = [JSON.parsefile(log_path) for log_path in DATALOGS]

    IND_MODELS = [
        "model$i" => [dict_log["individual_models"][i]["mae"] for dict_log in DICT_LOGS] for i = 1:T
    ]

    df = DataFrame([
        "dataset" => [split(ds, "/")[end] for ds in DATASETS],
        "size" => [dict_log["data_size"] for dict_log in DICT_LOGS],
        "K" => [dict_log["num_classes"] for dict_log in DICT_LOGS],
        IND_MODELS...,
        "MAE1" => [dict_log["ensemble_model_1"]["mae"] for dict_log in DICT_LOGS],
        "MAE2" => [dict_log["ensemble_model_2"]["mae"] for dict_log in DICT_LOGS],
        "STD1" => [dict_log["ensemble_model_1"]["std"] for dict_log in DICT_LOGS],
        "STD2" => [dict_log["ensemble_model_2"]["std"] for dict_log in DICT_LOGS],
        "ω1" => [dict_log["ensemble_model_1"]["weights"] for dict_log in DICT_LOGS],
        "ω2" => [dict_log["ensemble_model_2"]["weights"] for dict_log in DICT_LOGS],
        "time1" => [dict_log["ensemble_model_1"]["time"] for dict_log in DICT_LOGS],
        "time2" => [dict_log["ensemble_model_2"]["time"] for dict_log in DICT_LOGS]
    ])


    function row_formatter(val, i, j)
        # Placeholder for formatting rows if needed
        if typeof(val) <: Number
            nval = round(val, digits=4)
            return string(nval)
        elseif typeof(val) <: Array
            if length(val) == 2
                return "$(val[1]) x $(val[2])"
            else
                rounded_vals = [round(v, digits=3) for v in val]
                return "[" * join(rounded_vals, ", ") * "]"
            end
        else
            return val
        end
    end


    pretty_table(df; formatters=[row_formatter])

    file_suffix = ":$version:P$P:S$rnd_seed:T$T"

    results_md_path = "results/md/results$file_suffix.md"

    open(results_md_path, "w") do io
        write(io, "# Summary of Results\n\n")
        pretty_table_markdown_backend(io, df; formatters=[row_formatter])
    end


    aberto = stack(df[!, ["dataset", [x[1] for x in IND_MODELS]...]], variable_name=:model, value_name=:value)

    grouped = groupby(aberto, :dataset)

    resumen = combine(grouped, :value => (x -> mean(x)) => :mean_value,
        :value => (x -> std(x)) => :std_value)

    merged = leftjoin(resumen, df[!, Not([x[1] for x in IND_MODELS])], on=:dataset)

    means_md_path = "results/md/means$file_suffix.md"

    open(means_md_path, "w") do io
        write(io, "# Summary of Results\n\n")
        write(io, "version=$version, P=$P, rnd_seed=$rnd_seed, T=$T. Carrizosa model is 1, ours is 2. \n\n")
        write(io, "<p id=\"results\"></p>\n")
        pretty_table_markdown_backend(io, merged; formatters=[row_formatter])
    end


    function md_table_to_html(md_file)
        convert_command = `pandoc -i $md_file -o temp.html`
        run(convert_command)
        html_content = read("temp.html", String)
        html_template = read("sumary-template.html", String)
        final_html = replace(html_template, "{{TABLE_CONTENT}}" => html_content)
        new_file_path = replace(md_file, ".md" => ".html")
        open(new_file_path, "w") do io
            write(io, final_html)
        end
        new_file_name = split(new_file_path, "/")[end]    
        html_file_path = "results/html/$new_file_name"    
        mv(new_file_path, html_file_path, force=true) 
        return final_html
    end

    md_table_to_html(results_md_path)
    md_table_to_html(means_md_path)


    function fN(number)
        txt = string(round(number, digits=4))
        if number < 1
            txt = txt[2:end]  # Remove o "0" antes do ponto decimal'
        end
        return rpad(txt, 5, '0')  # Ajusta o espaçamento para números maiores
    end


    means_tex_path = "results/tex/means$file_suffix.tex"

    open(means_tex_path, "w") do io
        for row in eachrow(merged)
            mean = row.mean_value
            mean1 = row.MAE1
            mean2 = row.MAE2

            std = "\\small{$(fN(row.std_value))}"
            std1 = "\\small{$(fN(row.STD1))}"
            std2 = "\\small{$(fN(row.STD2))}"

            min_val = minimum([mean, mean1, mean2])

            # Formata mean
            mean_print = (mean <= min_val) ? "\\textbf{$(fN(mean))}" : "$(fN(mean))"

            # Formata mean1
            mean1_print = (mean1 <= min_val) ? "\\textbf{$(fN(mean1))}" : "$(fN(mean1))"

            # Formata mean2
            mean2_print = (mean2 <= min_val) ? "\\textbf{$(fN(mean2))}" : "$(fN(mean2))"

            rpads = [20, 10, 10, 10, 10, 10, 10, 10]
            contents = [
                rpad(row.dataset, rpads[1]),
                row.K,
                rpad(mean_print, rpads[2]),
                rpad(std, rpads[3]),
                rpad(mean1_print, rpads[4]),
                rpad(std1, rpads[5]),
                rpad(mean2_print, rpads[6]),
                rpad(std2, rpads[7])
            ]

            write(io, join(contents, " & ") * "\\\\ \\midrule") + write(io, "\n")
        end
    end

end
