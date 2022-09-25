using DFTK
using Dates
using JSON
using LinearAlgebra
using ForwardDiff
using TimerOutputs

include("./shifted_sternheimer.jl")

function run()

    # set up model
    a = 10.26
    lattice = a / 2 * [[0 1 1.];
                       [1 0 1.];
                       [1 1 0.]]
    Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))
    atoms     = [Si, Si]
    positions = [ones(3)/8, -ones(3)/8]
    kgrid = [1, 1, 1]
    Ecut = 5
    temperature = 0.0

    model = model_PBE(lattice, atoms, positions;
                      temperature, spin_polarization=:spinless)
    basis = PlaneWaveBasis(model; Ecut, kgrid)
    tol = 1e-12

    # log and results files
    res_file = "SCF_results.json"
    log_file_shifted = "sternheimer_log_shifted.json"
    log_file         = "sternheimer_log.json"

    # print setup
    println()
    println("hostname        = $(read(`hostname`, String))")
    println("started on      = $(Dates.now())")
    println("julia threads   = $(Threads.nthreads())")
    println("BLAS threads    = $(BLAS.get_num_threads())")
    println()
    println("a               = $(a)")
    println()
    println("temperature     = $(basis.model.temperature)")
    println("smearing        = $(basis.model.smearing)")
    println("lattice         = $(round.(basis.model.lattice, sigdigits=4))")
    println("Ecut            = $(basis.Ecut)")
    println("fft_size        = $(basis.fft_size)")
    println("kgrid           = $(basis.kgrid)")
    println("kshift          = $(basis.kshift)")
    println("n_irreducible_k = $(sum(length, basis.krange_allprocs))")
    println("n_bands         = $(DFTK.default_n_bands(basis.model))")
    println("n_electrons     = $(basis.model.n_electrons)")
    flush(stdout)
    println("\n--------------------------------")

    # results dictionnary
    res_dict = Dict(
                    "a"            => a,
                    "temperature"  => basis.model.temperature,
                    "smearing"     => string(basis.model.smearing),
                    "lattice"      => basis.model.lattice,
                    "Ecut"         => basis.Ecut,
                    "fft_size"     => basis.fft_size,
                    "kgrid"        => basis.kgrid,
                    "n_kirred"     => sum(length, basis.krange_allprocs),
                    "n_bands"      => DFTK.default_n_bands(basis.model),
                    "n_electr"     => basis.model.n_electrons,
                    #
                    "energies"     => Float64[],
                    "ndiag"        => Int[],
                    "α"            => Float64[],
                   )

    # custom callback for SCF solver
    function ExtractCallback()
        function callback(info)
            if info.stage != :finalize
                push!(res_dict["energies"], info.energies.total)
                push!(res_dict["ndiag"], length(info.diagonalization))
                push!(res_dict["α"], info.α)
            end
        end
    end
    callback_SCF = ExtractCallback() ∘ DFTK.ScfDefaultCallback()

    # custom callback for sternheimer solver
    log_dict_shifted = Dict{Int64, Array{Float64}}()
    log_dict         = Dict{Int64, Array{Float64}}()
    function callback_sternheimer!(dict)
        function callback(info)
            dict[info.n] = info.ch[:resnorm]
        end
    end

    # run SCF and sternheimer
    scfres = self_consistent_field(basis; tol=tol, callback=callback_SCF)
    # generate appropriate right-hand-side from atomic displacements
    R = [zeros(3), ones(3)]
    function Vψ(ε)
        T = typeof(ε)
        pos = positions + ε*R
        modelV = Model(Matrix{T}(lattice), atoms, pos; model_name="potential",
                       terms=[DFTK.AtomicLocal(),
                              DFTK.AtomicNonlocal()])
        basisV = PlaneWaveBasis(modelV; Ecut, kgrid)
        jambon = Hamiltonian(basisV)
        jambon * Vector{Matrix{Complex{T}}}(scfres.ψ)
    end
    δVψ = ForwardDiff.derivative(Vψ, 0.0)
    # default setup with extra bands
    println("\n--------------------------------")
    println("apply_χ0 with extra bands")
    DFTK.reset_timer!(DFTK.timer)
    δψ1 = DFTK.apply_χ0_4P(scfres.ham, scfres.ψ, scfres.occupation, scfres.εF,
                           scfres.eigenvalues, δVψ; scfres.occupation_threshold,
                           callback=callback_sternheimer!(log_dict))
    println(DFTK.timer)
    println("\n--------------------------------")
    println("apply_χ0 with shilfted hamiltonian")
    # setup without extra bands
    DFTK.reset_timer!(DFTK.timer)
    ψ_occ, occ_occ = DFTK.select_occupied_orbitals(basis,
                                                   scfres.ψ,
                                                   scfres.occupation;
                                                   threshold=scfres.occupation_threshold)
    ε_occ = [scfres.eigenvalues[ik][1:size(ψk,2)] for (ik, ψk) in enumerate(ψ_occ)]
    δψ2 = apply_χ0_shifted(scfres.ham, ψ_occ, occ_occ, scfres.εF, ε_occ, δVψ;
                           scfres.occupation_threshold,
                           callback=callback_sternheimer!(log_dict_shifted))
    println(DFTK.timer)

    println("\n--------------------------------")
    δρ1 = DFTK.compute_δρ(basis, scfres.ψ, δψ1, scfres.occupation)
    δρ2 = DFTK.compute_δρ(basis, ψ_occ, δψ2, occ_occ)
    @show norm(δρ1 - δρ2)
    @infiltrate

    # write output files
    res_dict["eigenvalues"] = scfres.eigenvalues
    res_dict["residuals"] = scfres.diagonalization[1].residual_norms
    open(fp -> JSON.print(fp, res_dict), res_file, "w")
    open(fp -> JSON.print(fp, log_dict_shifted), log_file_shifted, "w")
    open(fp -> JSON.print(fp, log_dict), log_file, "w")
    nothing
end

open("scf.log", "w") do io
    redirect_stdout(io) do
        run()
    end
end

