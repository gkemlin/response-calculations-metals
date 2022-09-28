using DFTK
using JLD2
using DataFrames
using LaTeXStrings
using Latexify
using LinearAlgebra

# adaptive choice of the number of extra bands with particular focus on one k-point
# if N_start=nothing, we use as starting point the default N_extra used for the computations
function adaptive_choice_Nextra(system, target_ratio; ik_list=nothing, N_start=nothing)

    scfres = load_scfres("scfres.jld2")

    basis = scfres.basis
    isnothing(ik_list) && (ik_list = 1:length(basis.kpoints))

    # display table
    df = DataFrame()
    df[!, L"$k$-point"] = [L"N",
                           L"\text{default } N_{\rm ex}",
                           L"\text{suggested } N_{\rm ex}"]

    N_extra_max = 1
    for ik in ik_list
        println("\n==========================================================")
        println("ik = $ik")
        kpt = basis.kpoints[ik]
        H = scfres.ham.blocks[ik]
        occ = scfres.occupation[ik]
        occupation_threshold = scfres.occupation_threshold
        N = findfirst(x->x<occupation_threshold, occ) - 1

        # separate betwwen selected and extra bands
        ψ = scfres.ψ[ik][:,1:N]
        ε = scfres.eigenvalues[ik][1:N]
        N_extra_default = (length(occ) - N)
        N_extra = isnothing(N_start) ? N_extra_default : N_start
        ψ_extra = scfres.ψ[ik][:,N+1:N+N_extra]
        ε_extra = scfres.eigenvalues[ik][N+1:N+N_extra]

        # kinetic preconditioner
        prec = DFTK.PreconditionerTPA(basis, kpt)

        # adaptive choice of the number of extra bands
        ξ = √( (ε_extra[end]-ε[1]) / (ε_extra[end]-ε[N]) )
        println("\nStarting point")
        println("--------------")
        @show ξ
        @show N_extra
        df[!, "$ik"] = [N, N_extra_default, 0]
        println("\nStarting adaptive algorithm")
        println("---------------------------")
        while ξ > target_ratio

            # add random extra band, orhogonalized wrt to (ψ, ψ_extra)
            ϕ = -1 .+ 2 .* randn(size(ψ,1))
            ϕ = ϕ - ψ*(ψ'ϕ) - ψ_extra*(ψ_extra'ϕ)
            ϕ /= norm(ϕ)
            ψ_extra = [ψ_extra ϕ]

            # run LOBPCG until we reach the required tolerance
            res = DFTK.lobpcg_hyper(H, [ψ ψ_extra]; prec,
                                    tol = (ε_extra[end] - ε[N])/50)
            ε_extra  = res.λ[N+1:end]
            ψ_extra  = res.X[:,N+1:end]
            residual = res.residual_norms[end]
            N_extra = size(res.X[:,N+1:end],2)
            @show residual
            @show (ε_extra[end] - ε[N]) / 50

            ξ =  √( (ε_extra[end]-ε[1]) / (ε_extra[end]-ε[N]) )
            @show ε_extra[end]
            @show ξ
            @show N_extra
            println("")
        end
        println("\nResults")
        println("-------")
        N_extra = size(ψ_extra,2)
        N_extra_max = max(N_extra - N_extra_default, N_extra_max)
        @show ξ
        (N_extra > N_extra_default) && println("Suggestion: add $(N_extra - N_extra_default) extra bands for k-point $ik.")
        df[!, "$ik"][3] = N_extra
    end
    println("\nResults")
    println("-------")
    println("Suggest to add $N_extra_max extra bands")
    open("table_N_extra_$(system)_$(target_ratio).tex", "w") do file
        write(file, latexify(df, env=:table, latex=true, fmt=FancyNumberFormatter(3)))
    end
    nothing
end
