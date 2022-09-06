using DFTK
using Dates
using JSON
using LinearAlgebra
using ForwardDiff

# function that runs SCF for silicon with a given lattice constant
function run(a::Float64, dir::String; α=0.8)

    # set up model
    lattice = a / 2 * [[0 1 1.];
                       [1 0 1.];
                       [1 1 0.]]
    Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))
    atoms     = [Si, Si]
    positions = [ones(3)/8, -ones(3)/8]
    kgrid = [1, 1, 1]
    Ecut = 50

    model = model_PBE(lattice, atoms, positions)
    basis = PlaneWaveBasis(model; Ecut, kgrid)
    tol = 1e-12

    # log and results files
    res_file = "SCF_results.json"
    log_file_noextra = "sternheimer_log_noextra.json"
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
    log_dict_noextra = Dict{Int64, Array{Float64}}()
    log_dict         = Dict{Int64, Array{Float64}}()
    function callback_sternheimer!(dict)
        function callback(info)
            dict[info.n] = info.ch[:resnorm]
        end
    end

    # run SCF and sternheimer
    scfres = self_consistent_field(basis; tol=tol, callback=callback_SCF,
                                   damping=α)
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
    δψ = DFTK.apply_χ0_4P(scfres.ham, scfres.ψ, scfres.occupation, scfres.εF,
                          scfres.eigenvalues, δVψ; scfres.occupation_threshold,
                          callback=callback_sternheimer!(log_dict))
    println(DFTK.timer)
    println("\n--------------------------------")
    println("apply_χ0 without extra bands")
    # setup without extra bands
    ψ_occ = [ψk[:, 1:4] for ψk in scfres.ψ]
    ε_occ = [εk[1:4] for εk in scfres.eigenvalues]
    occ   = [occk[1:4] for occk in scfres.occupation]
    DFTK.reset_timer!(DFTK.timer)
    δψ = DFTK.apply_χ0_4P(scfres.ham, ψ_occ, occ, scfres.εF, ε_occ, δVψ;
                          scfres.occupation_threshold,
                          callback=callback_sternheimer!(log_dict_noextra))
    println(DFTK.timer)

    # write output files
    res_dict["eigenvalues"] = scfres.eigenvalues
    res_dict["residuals"] = scfres.diagonalization[1].residual_norms
    open(fp -> JSON.print(fp, res_dict), res_file, "w")
    open(fp -> JSON.print(fp, log_dict_noextra), log_file_noextra, "w")
    open(fp -> JSON.print(fp, log_dict), log_file, "w")
    nothing
end

# run calculations for a range of lattice constants
a_list = LinRange(10, 11.30, 15)
for a in a_list

    rounded_a = round(a, sigdigits=5)

    if !isdir("silicon_a$(rounded_a)")
        mkdir("silicon_a$(rounded_a)")
    end
    dir = joinpath(@__DIR__, "silicon_a$(rounded_a)")

    cd(dir)
    open("scf_a$(rounded_a).log", "w") do io
        redirect_stdout(io) do
            run(a, dir)
        end
    end
    cd("../")
end
a_list = LinRange(11.31, 11.403, 15)
for a in a_list

    rounded_a = round(a, sigdigits=5)

    if !isdir("silicon_a$(rounded_a)")
        mkdir("silicon_a$(rounded_a)")
    end
    dir = joinpath(@__DIR__, "silicon_a$(rounded_a)")

    cd(dir)
    open("scf_a$(rounded_a).log", "w") do io
        redirect_stdout(io) do
            run(a, dir; α=0.6)
        end
    end
    cd("../")
end
