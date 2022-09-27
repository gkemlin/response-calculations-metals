using DFTK
using Dates
using JSON
using LinearAlgebra
using ForwardDiff
using JLD2

using MPI
disable_threading()

function run_scf()
    function attach_pseudos(atoms; pseudomap...)
        pseudomap = Dict(pseudomap)
        map(atoms) do (element)
            pspfile = get(pseudomap, element.symbol, nothing)
            ElementPsp(element.symbol, psp=load_psp(pspfile))
        end
    end

    lattice   = load_lattice("Al40.json")
    positions = load_positions("Al40.json")
    atoms     = load_atoms("Al40.json")
    atoms     = attach_pseudos(atoms; Al="hgh/pbe/al-q3.hgh")

    Ecut = 40
    kgrid = [3, 3, 1]
    temperature = 0.001
    tol = 1e-12
    mixing = KerkerMixing()

    model = model_PBE(lattice, atoms, positions; temperature)
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
        println("mixing          = $(mixing)")
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
                    "mixing"       => string(mixing),
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
    scfres = self_consistent_field(basis; tol, mixing, occupation_threshold=1e-8,
                                   callback=callback_SCF)
    if mpi_master()
        println(DFTK.timer)
    end
    save_scfres("scfres.jld2", scfres);
    R = [zeros(3) for pos in positions]
    for iR in 1:length(R)
        R[iR] = -ones(3) + 2*rand(3)
    end
    # generate appropriate right-hand-side from atomic displacements
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
    if mpi_master()
        println("\n--------------------------------")
        println("apply_χ0 with extra bands")
    end
    DFTK.reset_timer!(DFTK.timer)
    δψ = DFTK.apply_χ0_4P(scfres.ham, scfres.ψ, scfres.occupation, scfres.εF,
                          scfres.eigenvalues, δVψ; scfres.occupation_threshold,
                          callback=callback_sternheimer!(log_dict))
    if mpi_master()
        println(DFTK.timer)
        println("\n--------------------------------")
        println("apply_χ0 without extra bands")
    end
    # no extra bands
    DFTK.reset_timer!(DFTK.timer)
    ψ_occ, occ_occ = DFTK.select_occupied_orbitals(basis,
                                                   scfres.ψ,
                                                   scfres.occupation;
                                                   threshold=scfres.occupation_threshold)
    ε_occ = [scfres.eigenvalues[ik][1:size(ψk,2)] for (ik, ψk) in enumerate(ψ_occ)]
    δψ = DFTK.apply_χ0_4P(scfres.ham, ψ_occ, occ_occ, scfres.εF, ε_occ, δVψ;
                          scfres.occupation_threshold,
                          callback=callback_sternheimer!(log_dict_noextra))
    if mpi_master()
        println(DFTK.timer)
    end

    # write output files
    kpts       = DFTK.gather_kpts(scfres.basis.kpoints, scfres.basis)
    ε          = DFTK.gather_kpts(scfres.eigenvalues, scfres.basis)
    residuals  = DFTK.gather_kpts(scfres.diagonalization[1].residual_norms, scfres.basis)
    occupation = DFTK.gather_kpts(scfres.occupation, scfres.basis)
    if mpi_master()
        res_dict["eigenvalues"] = ε
        res_dict["residuals"]   = residuals
        res_dict["occupation"]  = occupation
        res_dict["kpts"]        = kpts
        res_dict["occupation_threshold"] = scfres.occupation_threshold
        open(fp -> JSON.print(fp, res_dict), res_file, "w")
    end
    open(fp -> JSON.print(fp, log_dict), log_file, "w")
    open(fp -> JSON.print(fp, log_dict_noextra), log_file_noextra, "w")
    nothing
end

# run scf if no log file
open("scf.log", "w") do io
    redirect_stdout(io) do
        run_scf()
    end
end

