# common functions to apply χ0, with the Schur and the direct method
using LinearMaps
using IterativeSolvers

# Solves (1-P) (H-εn) (1-P) δψn = - (1-P) rhs
# where 1-P is the projector on the orthogonal of ψk
# n is used for the preconditioning with ψk[:,n] and the optional callback
# /!\ It is assumed (and not checked) that ψk'Hk*ψk = Diagonal(εk) (extra states
# included).
function sternheimer_solver(Hk, ψk, εnk, rhs, n; callback=info->nothing,
                            ψk_extra=zeros(size(ψk,1), 0), εk_extra=zeros(0),
                            abstol=1e-9, reltol=0, verbose=false)
    basis = Hk.basis
    kpoint = Hk.kpoint

    # We use a Schur decomposition of the orthogonal of the occupied states
    # into a part where we have the partially converged, non-occupied bands
    # (which are Rayleigh-Ritz wrt to Hk) and the rest.

    # Projectors:
    # projector onto the computed and converged states
    P(ϕ) = ψk * (ψk' * ϕ)
    # projector onto the computed but nonconverged states
    P_extra(ϕ) = ψk_extra * (ψk_extra' * ϕ)
    # projector onto the computed (converged and unconverged) states
    P_computed(ϕ) = P(ϕ) + P_extra(ϕ)
    # Q = 1-P is the projector onto the orthogonal of converged states
    Q(ϕ) = ϕ - P(ϕ)
    # R = 1-P_computed is the projector onto the orthogonal of computed states
    R(ϕ) = ϕ - P_computed(ϕ)

    # We put things into the form
    # δψkn = ψk_extra * αkn + δψknᴿ ∈ Ran(Q)
    # where δψknᴿ ∈ Ran(R).
    # Note that, if ψk_extra = [], then 1-P = 1-P_computed and
    # δψkn = δψknᴿ is obtained by inverting the full Sternheimer
    # equations in Ran(Q) = Ran(R)
    #
    # This can be summarized as the following:
    #
    # <---- P ----><------------ Q = 1-P -----------------
    #              <-- P_extra -->
    # <--------P_computed -------><-- R = 1-P_computed ---
    # |-----------|--------------|------------------------
    # 1     N_occupied  N_occupied + N_extra

    # ψk_extra are not converged but have been Rayleigh-Ritzed (they are NOT
    # eigenvectors of H) so H(ψk_extra) = ψk_extra' (Hk-εn) ψk_extra should be a
    # real diagonal matrix.
    H(ϕ) = Hk * ϕ - εnk * ϕ
    ψk_exHψk_ex = Diagonal(real.(εk_extra .- εnk))

    # 1) solve for δψknᴿ
    # ----------------------------
    # writing αkn as a function of δψknᴿ, we get that δψknᴿ
    # solves the system (in Ran(1-P_computed))
    #
    # R * (H - εn) * (1 - M * (H - εn)) * R * δψknᴿ = R * (1 - M) * b
    #
    # where M = ψk_extra * (ψk_extra' (H-εn) ψk_extra)^{-1} * ψk_extra'
    # is defined above and b is the projection of -rhs onto Ran(Q).
    #
    b = -Q(rhs)
    bb = R(b -  H(ψk_extra * (ψk_exHψk_ex \ ψk_extra'b)))
    function RAR(ϕ)
        Rϕ = R(ϕ)
        Hψk_extra = H(ψk_extra)
        # Schur complement of (1-P) (H-εn) (1-P)
        # with the splitting Ran(1-P) = Ran(P_extra) ⊕ Ran(R)
        R(H(Rϕ)) - R(Hψk_extra) * (ψk_exHψk_ex \ Hψk_extra'Rϕ)
    end
    precon = PreconditionerTPA(basis, kpoint)
    DFTK.precondprep!(precon, ψk[:, n])
    function R_ldiv!(x, y)
        x .= R(precon \ R(y))
    end
    J = LinearMap{eltype(ψk)}(RAR, size(Hk, 1))
    δψknᴿ, ch = cg(J, bb; Pl=DFTK.FunctionPreconditioner(R_ldiv!), abstol, reltol,
                   verbose, log=true)
    info = (; basis=basis, kpoint=kpoint, ch=ch, n=n)
    callback(info)

    # 2) solve for αkn now that we know δψknᴿ
    # Note that αkn is an empty array if there is no extra bands.
    αkn = ψk_exHψk_ex \ ψk_extra' * (b - H(δψknᴿ))

    δψkn = ψk_extra * αkn + δψknᴿ
end

function compute_αmn(fm, fn, ratio)
    ratio == 0 && return ratio
    ratio * fn / (fn^2 + fm^2)
end


@views DFTK.@timing function apply_χ0_schur(ham, ψ, occ, εF, eigenvalues, δHψ;
                                            occupation_threshold, kwargs_sternheimer...)
    basis  = ham.basis
    model = basis.model
    temperature = model.temperature
    filled_occ = DFTK.filled_occupation(model)
    T = eltype(basis)
    Nk = length(basis.kpoints)

    # We first select orbitals with occupation number higher than
    # occupation_threshold for which we compute the associated response δψn,
    # the others being discarded to ψ_extra / ε_extra.
    # We then use the extra information we have from these additional bands,
    # non-necessarily converged, to split the sternheimer_solver with a Schur
    # complement.

    mask_occ   = map(occk -> isless.(occupation_threshold, occk), occ)
    mask_extra = map(occk -> (!isless).(occupation_threshold, occk), occ)

    ψ_occ   = [ψ[ik][:, maskk] for (ik, maskk) in enumerate(mask_occ)]
    ψ_extra = [ψ[ik][:, maskk] for (ik, maskk) in enumerate(mask_extra)]

    ε_occ   = [eigenvalues[ik][maskk] for (ik, maskk) in enumerate(mask_occ)]
    ε_extra = [eigenvalues[ik][maskk] for (ik, maskk) in enumerate(mask_extra)]

    occ_occ = [occ[ik][maskk] for (ik, maskk) in enumerate(mask_occ)]

    # First compute δεF
    δεF = zero(T)
    δocc = [zero(occ_occ[ik]) for ik = 1:Nk]  # = fn' * (δεn - δεF)
    if temperature > 0
        # First compute δocc without self-consistent Fermi δεF
        D = zero(T)
        for ik = 1:Nk, (n, εnk) in enumerate(ε_occ[ik])
            enred = (εnk - εF) / temperature
            δεnk = real(dot(ψ_occ[ik][:, n], δHψ[ik][:, n]))
            fpnk = (filled_occ
                    * Smearing.occupation_derivative(model.smearing, enred)
                    / temperature)
            δocc[ik][n] = δεnk * fpnk
            D += fpnk * basis.kweights[ik]
        end
        # compute δεF
        D = DFTK.mpi_sum(D, basis.comm_kpts)  # equal to minus the total DOS
        δocc_tot = DFTK.mpi_sum(sum(basis.kweights .* sum.(δocc)), basis.comm_kpts)
        δεF = δocc_tot / D
        # recompute δocc
        for ik = 1:Nk, (n, εnk) in enumerate(ε_occ[ik])
            enred = (εnk - εF) / temperature
            fpnk = (filled_occ
                    * Smearing.occupation_derivative(model.smearing, enred)
                    / temperature)
            δocc[ik][n] -= fpnk * δεF
        end
    end

    # compute δψnk band per band
    δψ = zero.(ψ)
    for ik = 1:Nk
        ψk = ψ_occ[ik]
        δψk = δψ[ik]

        εk = ε_occ[ik]
        for n = 1:length(εk)
            fnk = filled_occ * Smearing.occupation(model.smearing, (εk[n]-εF) / temperature)

            # explicit contributions (nonzero only for temperature > 0)
            for m = 1:length(εk)
                fmk = filled_occ * Smearing.occupation(model.smearing, (εk[m]-εF) / temperature)
                ddiff = Smearing.occupation_divided_difference
                ratio = filled_occ * ddiff(model.smearing, εk[m], εk[n], εF, temperature)
                αmn = compute_αmn(fmk, fnk, ratio)  # fnk * αmn + fmk * αnm = ratio
                δψk[:, n] .+= ψk[:, m] .* αmn .* (dot(ψk[:, m], δHψ[ik][:, n]) * (n != m))
            end

            # Sternheimer contribution
            δψk[:, n] .+= sternheimer_solver(ham.blocks[ik], ψk, εk[n], δHψ[ik][:, n], n;
                                             ψk_extra=ψ_extra[ik], εk_extra=ε_extra[ik],
                                             kwargs_sternheimer...)
        end
    end

    # pad δoccupation
    δoccupation = zero.(occ)
    for (ik, maskk) in enumerate(mask_occ)
        δoccupation[ik][maskk] .+= δocc[ik]
    end

    # keeping zeros for extra bands to keep the output δψ with the same size
    # than the input ψ
    (; δψ, δoccupation, δεF)
end

