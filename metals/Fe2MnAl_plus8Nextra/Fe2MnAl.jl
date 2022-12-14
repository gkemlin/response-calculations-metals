using DFTK
using Dates
using JSON
using LinearAlgebra
using ForwardDiff
using JLD2
using Random

using MPI
disable_threading()

include("../../apply_chi0.jl")

function run_scf()
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
                    "??"            => Float64[],
                   )

    # custom callback for SCF solver
    function ExtractCallback()
        function callback(info)
            if info.stage != :finalize
                push!(res_dict["energies"], info.energies.total)
                push!(res_dict["ndiag"], length(info.diagonalization))
                push!(res_dict["??"], info.??)
            end
        end
    end
    callback_SCF = ExtractCallback() ??? DFTK.ScfDefaultCallback()

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
    ??0 = guess_density(basis, magnetic_moments)
    scfres = DFTK.scf_potential_mixing(basis; tol, mixing, ??=??0,
                                       occupation_threshold=1e-8,
                                       n_ep_extra=11,
                                       callback=callback_SCF)
    if mpi_master()
        println(DFTK.timer)
    end
    save_scfres("scfres.jld2", scfres);
    # generate appropriate right-hand-side from atomic displacements
    R = [zeros(3) for pos in positions]
    Random.seed!(1234)
    for iR in 1:length(R)
        R[iR] = -ones(3) + 2*rand(3)
    end
    function V??(??)
        T = typeof(??)
        pos = positions + ??*R
        modelV = Model(Matrix{T}(lattice), atoms, pos; model_name="potential",
                       magnetic_moments,
                       terms=[DFTK.AtomicLocal(), DFTK.AtomicNonlocal()])
        basisV = PlaneWaveBasis(modelV; Ecut, kgrid)
        jambon = Hamiltonian(basisV)
        jambon * Vector{Matrix{Complex{T}}}(scfres.??)
    end
    ??V?? = ForwardDiff.derivative(V??, 0.0)
    # default setup with extra bands
    if mpi_master()
        println("\n--------------------------------")
        println("apply_??0 with extra bands")
    end
    DFTK.reset_timer!(DFTK.timer)
    (; ????, ??occupation, ????F) = apply_??0_schur(scfres.ham, scfres.??, scfres.occupation, scfres.??F,
                                              scfres.eigenvalues, ??V??; scfres.occupation_threshold,
                                              callback=callback_sternheimer!(log_dict))
    if mpi_master()
        println(DFTK.timer)
        println("\n--------------------------------")
        println("apply_??0 without extra bands")
    end
    # no extra bands
    DFTK.reset_timer!(DFTK.timer)
    ??_occ, occ_occ = DFTK.select_occupied_orbitals(basis,
                                                   scfres.??,
                                                   scfres.occupation;
                                                   threshold=scfres.occupation_threshold)
    ??_occ = [scfres.eigenvalues[ik][1:size(??k,2)] for (ik, ??k) in enumerate(??_occ)]
    (; ????, ??occupation, ????F) = apply_??0_schur(scfres.ham, ??_occ, occ_occ, scfres.??F, ??_occ, ??V??;
                                              scfres.occupation_threshold,
                                              callback=callback_sternheimer!(log_dict_noextra))
    if mpi_master()
        println(DFTK.timer)
    end

    # write output files
    kpts       = DFTK.gather_kpts(scfres.basis.kpoints, scfres.basis)
    ??          = DFTK.gather_kpts(scfres.eigenvalues, scfres.basis)
    residuals  = DFTK.gather_kpts(scfres.diagonalization[1].residual_norms, scfres.basis)
    occupation = DFTK.gather_kpts(scfres.occupation, scfres.basis)
    if mpi_master()
        res_dict["eigenvalues"] = ??
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

open("scf.log", "w") do io
    redirect_stdout(io) do
        run_scf()
    end
end

