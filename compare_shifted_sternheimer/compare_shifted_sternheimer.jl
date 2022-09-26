using DFTK
using Dates
using JSON
using LinearAlgebra
using ForwardDiff
using JLD2

using MPI
disable_threading()

include("./shifted_sternheimer.jl")

function run()
    function build_magnetic_moments(atoms::AbstractArray; magmoms...)
        magmoms = Dict(magmoms)
        map(atoms) do (element)
            magmoms[element.symbol]
        end
    end

    function attach_pseudos(atoms::AbstractArray; pseudomap...)
        pseudomap = Dict(pseudomap)

        map(atoms) do (element)
            pspfile = get(pseudomap, element.symbol, nothing)
            ElementPsp(element.symbol, psp=load_psp(pspfile))
        end
    end

    lattice   = load_lattice("AlFe2Mn_qe.in")
    positions = load_positions("AlFe2Mn_qe.in")
    atoms     = load_atoms("AlFe2Mn_qe.in")
    atoms     = attach_pseudos(atoms, Mn="hgh/pbe/mn-q15.hgh",
                               Fe="hgh/pbe/fe-q16.hgh",
                               Al="hgh/pbe/al-q3.hgh")
    magnetic_moments = build_magnetic_moments(atoms; Al=0.0, Fe=5.0, Mn=5.0)
    if mpi_master()
        println(atoms)
        println(magnetic_moments)
    end
    smearing         = Smearing.Gaussian()
    temperature      = 0.01
    Ecut             = 45
    kgrid            = [13, 13, 13]
    supersampling    = 2.0
    mixing           = KerkerMixing()
    tol              = 1e-12

    model = model_PBE(lattice, atoms, positions;
                      temperature, magnetic_moments, smearing)
    basis = PlaneWaveBasis(model; Ecut, kgrid)

    # log and results files
    proc_id  = MPI.Comm_rank(basis.comm_kpts)
    res_file = "SCF_results.json"
    log_file         = "sternheimer_log_proc$(proc_id).json"
    log_file_noextra = "sternheimer_log_proc$(proc_id)_noextra.json"

    # print setup
    if mpi_master()
        println()
        println("hostname        = $(read(`hostname`, String))")
        println("started on      = $(Dates.now())")
        println("julia threads   = $(Threads.nthreads())")
        println("BLAS threads    = $(BLAS.get_num_threads())")
        println("MPI procs       = $(mpi_nprocs())")
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
    end

    # results dictionnary
    res_dict = Dict(
                    "temperature"  => basis.model.temperature,
                    "smearing"     => string(basis.model.smearing),
                    "lattice"      => basis.model.lattice,
                    "Ecut"         => basis.Ecut,
                    "fft_size"     => basis.fft_size,
                    "kgrid"        => basis.kgrid,
                    "n_kirred"     => sum(length, basis.krange_allprocs),
                    "n_bands"      => DFTK.default_n_bands(basis.model),
                    "n_electr"     => basis.model.n_electrons,
                    "mpi_n_procs"  => DFTK.mpi_nprocs(),
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
    log_dict         = Dict{Tuple{Array{Float64}, Int64}, Dict{Int64, Array{Float64}}}()
    log_dict_noextra = Dict{Tuple{Array{Float64}, Int64}, Dict{Int64, Array{Float64}}}()
    function callback_sternheimer!(dict)
        function callback(info)
            coord   = info.kpoint.coordinate
            spin    = info.kpoint.spin
            kpt_id  = (coord, spin)
            n       = info.n
            ch      = info.ch

            if haskey(dict, kpt_id)
                dict[kpt_id][n] = ch[:resnorm]
            else
                dict[kpt_id] = Dict{Int64, Array{Float64}}(n => ch[:resnorm])
            end
        end
    end
    # run SCF and sternheimer
    DFTK.reset_timer!(DFTK.timer)
    ρ0 = guess_density(basis, magnetic_moments)
    scfres = DFTK.scf_potential_mixing(basis; tol, mixing, ρ=ρ0,
                                       occupation_threshold=1e-8,
                                       callback=callback_SCF)
    if mpi_master()
        println(DFTK.timer)
    end
    save_scfres("scfres.jld2", scfres);
    # generate appropriate right-hand-side from atomic displacements
    R = [zeros(3) for pos in positions]
    for iR in 1:length(R)
        R[iR] = -ones(3) + 2*rand(3)
    end
    function Vψ(ε)
        T = typeof(ε)
        pos = positions + ε*R
        modelV = Model(Matrix{T}(lattice), atoms, pos; model_name="potential",
                       magnetic_moments,
                       terms=[DFTK.AtomicLocal(), DFTK.AtomicNonlocal()])
        basisV = PlaneWaveBasis(modelV; Ecut, kgrid)
        jambon = Hamiltonian(basisV)
        jambon * Vector{Matrix{Complex{T}}}(scfres.ψ)
    end
    δVψ = ForwardDiff.derivative(Vψ, 0.0)
    # default setup with extra bands
    if mpi_master()
        println("\n--------------------------------")
        println("apply_χ0 with extra bands")
    end
    DFTK.reset_timer!(DFTK.timer)
    δψ1 = DFTK.apply_χ0_4P(scfres.ham, scfres.ψ, scfres.occupation, scfres.εF,
                           scfres.eigenvalues, δVψ; scfres.occupation_threshold,
                           callback=callback_sternheimer!(log_dict))
    if mpi_master()
        println(DFTK.timer)
        println("\n--------------------------------")
        println("apply_χ0 without extra bands")
    end
    # no extra bands
    DFTK.reset_timer!(DFTK.timer)
    println("\n--------------------------------")
    println("apply_χ0 with shilfted hamiltonian")
    # setup with the shifted Hamiltonian
    DFTK.reset_timer!(DFTK.timer)
    ψ_occ, occ_occ = DFTK.select_occupied_orbitals(basis,
                                                   scfres.ψ,
                                                   scfres.occupation;
                                                   threshold=scfres.occupation_threshold)
    ε_occ = [scfres.eigenvalues[ik][1:size(ψk,2)] for (ik, ψk) in enumerate(ψ_occ)]
    δψ2 = apply_χ0_shifted(scfres.ham, ψ_occ, occ_occ, scfres.εF, ε_occ, δVψ;
                           scfres.occupation_threshold, abstol=1e-12,
                           callback=callback_sternheimer!(log_dict_shifted))
    println(DFTK.timer)

    println("\n--------------------------------")
    δψ1, _ = DFTK.select_occupied_orbitals(basis,
                                           δψ1,
                                           scfres.occupation;
                                           threshold=scfres.occupation_threshold)
    δρ1 = DFTK.compute_δρ(basis, ψ_occ, δψ1, occ_occ)
    δρ2 = DFTK.compute_δρ(basis, ψ_occ, δψ2, [ones(size(occk)) for occk in occ_occ])

    @show norm(δρ1 - δρ2)
    @show sum(δρ1)
    @show sum(δρ2)

    # make sure we have the same density response
    @assert norm(δρ1 - δρ2) < 5e-8
    @assert sum(δρ1) < 1e-12
    @assert sum(δρ2) < 1e-12

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

