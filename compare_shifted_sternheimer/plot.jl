using PGFPlots
using LaTeXStrings
using JSON
using LinearAlgebra
using DataFrames
using Latexify

pushPGFPlotsPreamble("\\usepackage{amsmath}")

# colors for plots
define_color("myred",    [228,26,28])
define_color("myblue",   [55,124,184])
define_color("mygreen",  [77,175,74])
define_color("myorange", [255,127,14])
define_color("myviolet", [148,103,189])
mycolors=["myred", "myblue", "mygreen", "myorange", "myviolet"]
style = "thick"

# plot table and convergence graph for every k-point with index in ik_list
function plot_cvg(system::String, ik_list; spin=false)
    res = open(JSON.parse, "SCF_results.json")
    ε          = res["eigenvalues"]
    residuals  = res["residuals"]
    occupation = res["occupation"]
    kpts       = res["kpts"]
    threshold  = res["occupation_threshold"]
    n_procs    = res["mpi_n_procs"]

    # post-process results
    N_list   = findfirst.(x->x<threshold, occupation)
    gap_list = [εk[N_list[ik]] - εk[N_list[ik]-1] for (ik, εk) in enumerate(ε)]

    display(hcat(gap_list...))
    display(hcat([[ε[i] occupation[i] residuals[i]] for i in ik_list]...))

    # display table
    df = DataFrame()
    df[!, L"$k$-point"] = [L"N",
                           L"\varepsilon_{N-2}| f_{N-2}",
                           L"\varepsilon_{N-1}| f_{N-1}",
                           L"\varepsilon_N    | f_N",
                           L"\varepsilon_{N+1}| f_{N+1}",
                           L"\varepsilon_{N+1} - \varepsilon_N",
                           L"\#iterations $n=N$ Schur",
                           L"\#iterations $n=N$ shifted"]

    # span selected k-points
    for ik in ik_list
        N = N_list[ik]-1
        df[!, "$ik"] = vcat(N,                # number of "occupied" states
                            ε[ik][N-2:N+1],   # energies around the occupation threshold
                            gap_list[ik],     # gap
                            0, 0)
        df[!, "$ik bis"] = vcat("",
                                # occupatio around the occupation threshold
                                occupation[ik][N-2:N+1],
                                "", "", "")
        kpt_id = (Float64.(kpts[ik]["coordinate"]), kpts[ik]["spin"])
        key = "$(kpt_id)"
        println(key)
        for id in 0:(n_procs-1)
            open("sternheimer_log_proc$(id).json", "r") do file
                open("sternheimer_log_proc$(id)_shifted.json", "r") do file_shifted
                    dict = JSON.parse(file)
                    dict_shifted = JSON.parse(file_shifted)
                    if key in keys(dict)
                        # update DataFrame
                        max_schur   = maximum(length.(values(dict[key])))
                        max_noschur = maximum(length.(values(dict_shifted[key])))
                        df[!, "$ik"][end-1] = max_schur
                        df[!, "$ik"][end]   = max_noschur

                        # plot iterations
                        kpt_short = round.(kpt_id[1], digits=3)
                        p = Plots.Linear[]
                        n_list = sort(parse.(Int64, keys(dict[key])))
                        for n in n_list
                            if n==1
                                color = mycolors[3]
                                push!(p, Plots.Linear(Float64.(dict[key]["$n"]),
                                                      legendentry=L"Schur $n=%$(n)$",
                                                      style="solid, thick, $color, mark repeat=5", mark="triangle"))
                                push!(p, Plots.Linear(Float64.(dict_shifted[key]["$n"]),
                                                      legendentry=L"shifted $n=%$(n)$",
                                                      style="dashed, thick, $color, mark repeat=5", mark="triangle"))
                            elseif n==div(3N,4)
                                color = mycolors[2]
                                push!(p, Plots.Linear(Float64.(dict[key]["$n"]),
                                                      legendentry=L"Schur $n=%$(n)$",
                                                      style="solid, thick, $color, mark repeat=5", mark="+"))
                                push!(p, Plots.Linear(Float64.(dict_shifted[key]["$n"]),
                                                      legendentry=L"shifted $n=%$(n)$",
                                                      style="dashed, thick, $color, mark repeat=5", mark="+"))
                            elseif n==N
                                color = mycolors[1]
                                push!(p, Plots.Linear(Float64.(dict[key]["$n"]),
                                                      legendentry=L"Schur $n=%$(n)$",
                                                      style="solid, thick, $color, mark repeat=5", mark="x"))
                                push!(p, Plots.Linear(Float64.(dict_shifted[key]["$n"]),
                                                      legendentry=L"shifted $n=%$(n)$",
                                                      style="dashed, thick, $color, mark repeat=5", mark="x"))
                            end
                        end
                        if spin
                            s = kpt_id[2] == 1 ? ", spin \$\\uparrow\$" : ", spin \$\\downarrow\$"
                        else
                            s = ""
                        end
                        g = Axis(p, title=L"$k$-point $%$(kpt_short)$%$(s)",
                                 xlabel="iterations",
                                 ylabel="residual", ymode="log",
                                 legendStyle="at={(1.05,0.5)}, anchor=west")
                        save("sternheimer_$(system)_alln_$(key).pdf", g)
                    end
                end
            end
        end
    end
    println("Convergence data for k-points $ik_list located at")
    display([kpts[ik]["coordinate"] for ik in ik_list])
    @show df
    open("table_$(system).tex", "w") do file
        write(file, latexify(df, env=:table, latex=true, fmt=FancyNumberFormatter(3)))
    end
    nothing
end

# function to plot convergence for all bands at each k-point
function all_plot(system::String; spin=false)
    res = open(JSON.parse, "SCF_results.json")
    n_procs = res["mpi_n_procs"]

    # span all proc to plot the convergence of the Sternheimer solver for every band
    # at each k-point
    for id in 0:(n_procs-1)
        open("sternheimer_log_proc$(id).json", "r") do file
            open("sternheimer_log_proc$(id)_shifted.json", "r") do file_shifted
                dict = JSON.parse(file)
                dict_shifted = JSON.parse(file_shifted)
                for key in keys(dict)
                    kpt_id = eval(Meta.parse(key))
                    kpt_short = round.(kpt_id[1], digits=3)
                    g = GroupPlot(2, 1, groupStyle="horizontal sep = 2cm")
                    p = Plots.Linear[] # plots with Schur
                    q = Plots.Linear[] # plots without Schur
                    i = 1 # color counter
                    n_list = sort(parse.(Int64, keys(dict[key])))
                    for n in n_list
                        color = mycolors[i]
                        i = i%length(mycolors) + 1
                        push!(p, Plots.Linear(Float64.(dict[key]["$n"]),
                                              style="solid, thick, $color", mark="x"))
                        push!(q, Plots.Linear(Float64.(dict_shifted[key]["$n"]),
                                              style="solid, thick, $color", mark="x"))
                    end
                    if spin
                        s = kpt_id[2] == 1 ? ", spin \$\\uparrow\$" : ", spin \$\\downarrow\$"
                    else
                        s = ""
                    end
                    push!(g, Axis(p, title=L"Schur -- $k$-point $%$(kpt_short)$%$(s)",
                                  xlabel="iterations",
                                  ylabel="residual", ymode="log",
                                  legendStyle="at={(0.95,0.95)}, anchor=north east"))
                    push!(g, Axis(q, title=L"shifted -- $k$-point $%$(kpt_short)$%$(s)",
                                  xlabel="iterations",
                                  ylabel="residual", ymode="log",
                                  legendStyle="at={(0.95,0.95)}, anchor=north east"))
                    save("all_plots/sternheimer_$(system)_$(key).pdf", g)
                end
            end
        end
    end
    nothing
end

# plot data for k-points with index in ik_list
ik_list = [96, 236]
plot_cvg("Fe2MnAl", ik_list, spin=true)

# uncomment to plot convergence for every k-points
# [for Heusler compounds it will generate 280 pdf !]
all_plot("Fe2MnAl", spin=true)
