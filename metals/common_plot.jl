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
                           L"\#iterations $n=N$ direct"]

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
                open("sternheimer_log_proc$(id)_noextra.json", "r") do file_noextra
                    dict = JSON.parse(file)
                    dict_noextra = JSON.parse(file_noextra)
                    if key in keys(dict)
                        # update DataFrame
                        max_schur   = maximum(length.(values(dict[key])))
                        max_noschur = maximum(length.(values(dict_noextra[key])))
                        df[!, "$ik"][end-1] = max_schur
                        df[!, "$ik"][end]   = max_noschur

                        # plot iterations
                        kpt_short = round.(kpt_id[1], digits=3)
                        g = GroupPlot(2, 1, groupStyle="horizontal sep = 2cm")
                        p = Plots.Linear[]
                        q = Plots.Linear[]
                        i = 1 # color counter
                        n_list = sort(parse.(Int64, keys(dict[key])))
                        for n in n_list
                            color = mycolors[i]
                            i = i%length(mycolors) + 1
                            push!(p, Plots.Linear(Float64.(dict[key]["$n"]),
                                                  style="solid, thick, $color", mark="x"))
                            push!(q, Plots.Linear(Float64.(dict_noextra[key]["$n"]),
                                                  style="solid, thick, $color", mark="x"))
                        end
                        if spin
                            s = kpt_id[2] == 1 ? " \$\\uparrow\$" : " \$\\downarrow\$"
                        else
                            s = ""
                        end
                        push!(g, Axis(p, title=L"Schur -- $k$-point at $%$(kpt_short)$%$(s)",
                                      xlabel="iterations",
                                      ylabel="residual", ymode="log",
                                      legendStyle="at={(0.95,0.95)}, anchor=north east"))
                        push!(g, Axis(q, title=L"direct -- $k$-point at $%$(kpt_short)$%$(s)",
                                      xlabel="iterations",
                                      ylabel="residual", ymode="log",
                                      legendStyle="at={(0.95,0.95)}, anchor=north east"))
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

function plot_ratios(system::String; spin=false, ξ=nothing)
    res = open(JSON.parse, "SCF_results.json")
    ε          = res["eigenvalues"]
    residuals  = res["residuals"]
    occupation = res["occupation"]
    kpts       = res["kpts"]
    threshold  = res["occupation_threshold"]
    n_procs    = res["mpi_n_procs"]

    color = "myblue"
    color_real = "mygreen"
    color_noextra = "myorange"

    # post-process results
    N_list   = findfirst.(x->x<threshold, occupation)

    # display table
    df = DataFrame()
    df[!, L"$k$-point"] = [L"N",
                           L"\#iterations $n=1$ Schur",
                           L"\#iterations $n=N$ Schur",
                           L"\#iterations $n=N$ direct"]

    spin_list = spin ? [1,2] : 1
    for s in spin_list
        p  = []
        p_noextra  = []
        pr = []
        for id in 0:(n_procs-1)
            open("sternheimer_log_proc$(id).json", "r") do file
                open("sternheimer_log_proc$(id)_noextra.json", "r") do file_noextra
                    dict = JSON.parse(file)
                    dict_noextra = JSON.parse(file_noextra)
                    for key in keys(dict)
                        kpt_id = eval(Meta.parse(key))
                        n_list = sort(parse.(Int64, keys(dict[key])))
                        ik = findfirst(x->(x["coordinate"],x["spin"])==kpt_id, kpts)
                        εk = ε[ik]
                        N = N_list[ik]-1
                        if kpt_id[2] == s
                            df[!, "$ik"] = [N, length(dict[key]["1"]),
                                            length(dict[key]["$N"]),
                                            length(dict_noextra[key]["$N"])]
                            cvg_ratio = length(dict[key]["$N"]) / length(dict[key]["1"])
                            cvg_ratio_noextra = length(dict_noextra[key]["$N"]) / length(dict_noextra[key]["1"])
                            theoretical_ratio = √( (εk[end] - εk[1]) / (εk[end] - εk[N]) )
                            push!(p, (cvg_ratio, ik))
                            push!(p_noextra, (cvg_ratio_noextra, ik))
                            push!(pr, (theoretical_ratio, ik))
                        end
                    end
                end
            end
        end
        pp = [Plots.Linear([x[2] for x in p_noextra], [x[1] for x in p_noextra],
                           legendentry="direct\\;\\;",
                           style="solid, thick, $color_noextra, only marks", mark="o")]
        push!(pp, Plots.Linear([x[2] for x in p], [x[1] for x in p],
                               legendentry="Schur\\;\\;",
                               style="solid, thick, $color, only marks", mark="square"))
        push!(pp, Plots.Linear([x[2] for x in pr], [x[1] for x in pr],
                               legendentry=L"\xi_{N_{\rm ex}}^l\;\;",
                               style="solid, thick, $color_real, only marks", mark="x"))
        if !isnothing(ξ)
            push!(pp, Plots.Linear([x[2] for x in p], [ξ for x in p],
                                   legendentry=L"\xi = %$ξ",
                                   style="solid, thick, $color_real", mark="none"))
        end
        ncol = isnothing(ξ) ? 3 : 4
        g = Axis(pp, xlabel=L"k\text{-points}", ymin=1.5, ymax=5.5,
                 legendStyle="at={(0.5,1.05)}, anchor=south, legend columns=$ncol")
        save("sternheimer_$(system)_ratios_$s.pdf", g)
    end
    open("table_$(system)_iterations.tex", "w") do file
        write(file, latexify(df, env=:table, latex=true, fmt=FancyNumberFormatter(3)))
    end
    nothing
end

# plot histogram of iterations per k point
function plot_histo(system::String)
    res = open(JSON.parse, "SCF_results.json")
    ε          = res["eigenvalues"]
    residuals  = res["residuals"]
    occupation = res["occupation"]
    kpts       = res["kpts"]
    threshold  = res["occupation_threshold"]
    n_procs    = res["mpi_n_procs"]

    color_up = "myblue"
    color_down = "mygreen"

    g = GroupPlot(2, 1, groupStyle="horizontal sep = 2cm")
    p = []
    q = []
    p_noextra = []
    q_noextra = []
    for id in 0:(n_procs-1)
        open("sternheimer_log_proc$(id).json", "r") do file
            open("sternheimer_log_proc$(id)_noextra.json", "r") do file_noextra
                dict = JSON.parse(file)
                dict_noextra = JSON.parse(file_noextra)
                for key in keys(dict)
                    kpt_id = eval(Meta.parse(key))
                    n_list = sort(parse.(Int64, keys(dict[key])))
                    ik = findfirst(x->(x["coordinate"],x["spin"])==kpt_id, kpts)
                    spin = kpt_id[2]
                    for n in n_list
                        if spin == 1
                            push!(p, (length(dict[key]["$n"]), ik))
                            push!(p_noextra, (length(dict_noextra[key]["$n"]), ik))
                        elseif spin == 2
                            push!(q, (length(dict[key]["$n"]), ik))
                            push!(q_noextra, (length(dict_noextra[key]["$n"]), ik))
                        end
                    end
                end
            end
        end
    end
    pp = [Plots.Linear([x[2] for x in p], [x[1] for x in p],
                       legendentry=L"\uparrow",
                       style="solid, thick, $color_up, only marks", mark="o")]
    push!(pp, Plots.Linear([x[2] for x in q], [x[1] for x in q],
                           legendentry=L"\downarrow",
                           style="solid, thick, $color_down, only marks", mark="x"))
    push!(g, Axis(pp, title="Schur",
                  xlabel=L"k\text{-points}",
                  ylabel="number of iterations", ymax=130))
    pp_noextra = [Plots.Linear([x[2] for x in p_noextra], [x[1] for x in p_noextra],
                               legendentry=L"\uparrow",
                               style="solid, thick, $color_up, only marks", mark="o")]
    push!(pp_noextra, Plots.Linear([x[2] for x in q_noextra], [x[1] for x in q_noextra],
                                   legendentry=L"\downarrow",
                                   style="solid, thick, $color_down, only marks", mark="x"))
    push!(g, Axis(pp_noextra, title="direct",
                  xlabel=L"k\text{-points}",
                  ylabel="number of iterations", ymax=130))
    save("sternheimer_$(system)_histo.pdf", g)
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
            open("sternheimer_log_proc$(id)_noextra.json", "r") do file_noextra
                dict = JSON.parse(file)
                dict_noextra = JSON.parse(file_noextra)
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
                        push!(q, Plots.Linear(Float64.(dict_noextra[key]["$n"]),
                                              style="solid, thick, $color", mark="x"))
                    end
                    if spin
                        s = kpt_id[2] == 1 ? " \$\\uparrow\$" : " \$\\downarrow\$"
                    else
                        s = ""
                    end
                    push!(g, Axis(p, title=L"Schur -- $k$-point at $%$(kpt_short)$%$(s)",
                                  xlabel="iterations",
                                  ylabel="residual", ymode="log",
                                  legendStyle="at={(0.95,0.95)}, anchor=north east"))
                    push!(g, Axis(q, title=L"direct -- $k$-point at $%$(kpt_short)$%$(s)",
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
