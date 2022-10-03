using PGFPlots
using LaTeXStrings
using JSON
using LinearAlgebra

pushPGFPlotsPreamble("\\usepackage{amsmath}")

# colors for plots
define_color("myred",    [228,26,28])
define_color("myblue",   [55,124,184])
define_color("mygreen",  [77,175,74])
define_color("myorange", [255,127,14])
define_color("myviolet", [148,103,189])
style = "thick"

# plot convergence of the last band
g = Axis(xlabel="gap", xmode="log",
         ylabel="iterations",
         legendStyle="at={(0.95,0.95)}, anchor=north east")

gap_list         = Float64[]
ite_list         = Int64[]
ite_noextra_list = Int64[]

for dir in readdir()
    if isdir(dir)
        res = open(JSON.parse, joinpath(dir, "SCF_results.json"))

        ε = res["eigenvalues"][1]
        push!(gap_list, ε[5] - ε[4])

        open(joinpath(dir, "sternheimer_log.json"), "r") do file
            dict = JSON.parse(file)
            push!(ite_list, length(dict["4"]))
        end
        open(joinpath(dir, "sternheimer_log_noextra.json"), "r") do file
            dict = JSON.parse(file)
            push!(ite_noextra_list, length(dict["4"]))
        end
    end
end
push!(g, Plots.Linear(gap_list, ite_list, legendentry="Schur",
                      style="thick, mygreen, mark size=4pt", mark="x"))
push!(g, Plots.Linear(gap_list, ite_noextra_list, legendentry="direct",
                      style="thick, myblue, mark size=4pt", mark="o"))
save("sternheimer_silicon_n4.pdf", g)
