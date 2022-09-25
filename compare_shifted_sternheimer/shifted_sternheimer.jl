# we implement here the solution of the sternheimer equation with a shifted
# Hamiltonian as it is presented in eq (72) (and with same notations) of
# S. Baroni, S. de Gironcoli, A. Dal Corso, and P. Giannozzi.
# Phonons and related crystal properties from density-functional perturbation theory.
# Reviews of Modern Physics, 73(2):515–562, 2001.

using LinearMaps
using IterativeSolvers

@views DFTK.@timing function apply_χ0_shifted(ham, ψ, occ, εF, eigenvalues, δHψ;
                                              occupation_threshold,
                                              callback=info->nothing,
                                              abstol=1e-9, reltol=0, verbose=false)
    basis  = ham.basis
    model = basis.model
    temperature = model.temperature
    filled_occ = DFTK.filled_occupation(model)
    Nk = length(basis.kpoints)
    T = eltype(basis)

    # First compute δεF
    δεF = zero(T)
    δocc = [zero(occ[ik]) for ik = 1:Nk]  # = fn' * (δεn - δεF)
    if temperature > 0
        # First compute δocc without self-consistent Fermi δεF
        D = zero(T)
        for ik = 1:Nk
            for (n, εnk) in enumerate(eigenvalues[ik])
                enred = (εnk - εF) / temperature
                δεnk = real(dot(ψ[ik][:, n], δHψ[ik][:, n]))
                fpnk = (filled_occ
                        * Smearing.occupation_derivative(model.smearing, enred)
                        / temperature)
                δocc[ik][n] = δεnk * fpnk
                D += fpnk * basis.kweights[ik]
            end
        end
        # compute δεF
        D = DFTK.mpi_sum(D, basis.comm_kpts)  # equal to minus the total DOS
        δocc_tot = DFTK.mpi_sum(sum(basis.kweights .* sum.(δocc)), basis.comm_kpts)
        δεF = δocc_tot / D
    end


    # compute δψnk band per band
    δψ = zero.(ψ)
    for ik = 1:Nk
        ψk = ψ[ik]
        δψk = δψ[ik]
        εk = eigenvalues[ik]
        Hk = ham.blocks[ik]
        δHψk = δHψ[ik]
        kpoint = basis.kpoints[ik]

        # build operator Q (independant of n)
        Δ = 4 * temperature
        A = Diagonal(εF .+ Δ .- εk)
        Q(ϕ) = ψk * A * (ψk' * ϕ)

        for n = 1:length(εk)
            θFn    = filled_occ * Smearing.occupation(model.smearing, (εk[n]-εF) / temperature)
            θFm(m) = filled_occ * Smearing.occupation(model.smearing, (εk[m]-εF) / temperature)
            θnm(m) = Smearing.occupation(model.smearing, (εk[m]-εk[n]) / temperature)
            θmn(m) = Smearing.occupation(model.smearing, (εk[n]-εk[m]) / temperature)
            ddiff = Smearing.occupation_divided_difference
            Rmn(m) = filled_occ * ddiff(model.smearing, εk[n], εk[m], εF, temperature)

            # build operator Pn (dependant of n)
            β = Diagonal([θFn*θnm(m) + θFm(m)*θmn(m) + A[m,m]*Rmn(m)*θmn(m)
                          for m in 1:length(εk)])
            Pn(ϕ) = ψk * β * (ψk' * ϕ)

            # build rhs
            δHψkn = δHψk[:,n]
            rhs = - (θFn*δHψkn - Pn(δHψkn))

            # build and solve shifted sternheimer equation
            function shifted_ham(ϕ)
                Hk*ϕ + Q(ϕ) - εk[n]*ϕ
            end
            precon = PreconditionerTPA(basis, kpoint)
            DFTK.precondprep!(precon, ψk[:, n])
            function ldiv!(x, y)
                x .= precon \ y
            end
            J = LinearMap{eltype(ψk)}(shifted_ham, size(Hk, 1))
            δψkn, ch = cg(J, rhs; Pl=DFTK.FunctionPreconditioner(ldiv!),
                          abstol, reltol, verbose, log=true)
            info = (; basis=basis, kpoint=kpoint, ch=ch, n=n)
            callback(info)
            δψk[:,n] = δψkn - Rmn(n) * δεF * ψk[:,n]
        end
        δψ[ik] = δψk
    end
    δψ
end

