using JSON
using DataFrames
using PrettyTables
using XLSX
using Statistics


DATASETS = Array{String,1}()
if isfile("datasets.txt")
    open("datasets.txt", "r") do f
        for line in eachline(f)
            line = strip(line)
            if !isempty(line) && !startswith(line, "#")
                push!(DATASETS, line)
            end
        end
    end
else
    # Se datasets.txt não existir, usar ARGS como fallback
    DATASETS = [ARGS[1]]
end

W = 1
version = "v3"
rnd_seed = 651
T = 15

DATALOGS = ["$data_path/testlog-$version-$W-$rnd_seed-$T.json" for data_path in DATASETS]

DICT_LOGS = [JSON.parsefile(log_path) for log_path in DATALOGS]

IND_MODELS = [
    "model$i" => [dict_log["individual_models"][i]["mae"] for dict_log in DICT_LOGS] for i = 1:7
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

open("summary_results-$version-$W-$rnd_seed-$T.md", "w") do io
    write(io, "# Summary of Results\n\n")
    pretty_table_markdown_backend(io, df; formatters=[row_formatter])
end


aberto = stack(df[!, ["dataset", [x[1] for x in IND_MODELS]...]], variable_name=:model, value_name=:value)

grouped = groupby(aberto, :dataset)

resumen = combine(grouped, :value => (x -> mean(x)) => :mean_value,
    :value => (x -> std(x)) => :std_value)

merged = leftjoin(resumen, df[!, Not([x[1] for x in IND_MODELS])], on=:dataset)

open("summary_results-means-$version-$W-$rnd_seed-$T.md", "w") do io
    write(io, "# Summary of Results\n\n")
    pretty_table_markdown_backend(io, merged; formatters=[row_formatter])
end


function md_table_to_html(md_file)
    convert_command = `pandoc -i $md_file -o temp.html`
    run(convert_command)
    html_content = read("temp.html", String)
    html_template = read("sumary-template.html", String)
    final_html = replace(html_template, "{{TABLE_CONTENT}}" => html_content)
    new_file_name = replace(md_file, ".md" => ".html")
    open(new_file_name, "w") do io
        write(io, final_html)
    end
    return final_html
end

md_table_to_html("summary_results-$version-$W-$rnd_seed-$T.md")
md_table_to_html("summary_results-means-$version-$W-$rnd_seed-$T.md")

XLSX.writetable("summary_results-means-$version-$W-$rnd_seed-$T.xlsx", merged[!, Not(:ω2, :ω1)], overwrite=true)


function fN(number)
    txt = string(round(number, digits=4))
    if number < 1
        txt = txt[2:end]  # Remove o "0" antes do ponto decimal'
    end
    return rpad(txt, 5, '0')  # Ajusta o espaçamento para números maiores
end


open("summary_results-means-formatted-$version-$W-$rnd_seed-$T.tex", "w") do io
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

move_command = `mv summary_results* results`
run(move_command)