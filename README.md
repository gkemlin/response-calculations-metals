Julia code performing the numerical simulation from our paper on robust response
calculations for metals, available on [arXiv](https://arxiv.org/abs/2210.04512).
The framework described in this paper is implemented by default in
[DFTK](https://dftk.org). For the metals, we run DFTK with MPI, see
[Timings and parallelization](https://docs.dftk.org/stable/tricks/parallelization/).

# Dependencies
Julia 1.7 with the following Julia libraries:
- [DFTK.jl](https://dftk.org) v0.5.5 for the simulation environment (this code
  might not work with more recent versions of DFTK);
- [ForwardDiff.jl](https://juliadiff.org/ForwardDiff.jl/stable/) for the
  generation of the perturbation `δV` from atomic displacements (with random
  seed fixed to reproduce the same results from the paper);
- MPI, LinearAlgebra, IterativeSolvers, Random for related computations;
- Dates, DataFrames, JLD2, JSON, PGFPlots, Latexify, LaTeXStrings, Plots,
  TimerOutputs for saving and reading results.

# Organisation
Numerical examples are gathered as follows:
- the silicon examples can be found in `silicon/`;
- the metallic examples can be found in `metals/system` where system is either
  `Al40` or `Fe2MnAl`, with default extra bands and with number of extra bands
  suggested by the adaptive algorithm from the appendix;
- the comparision with a shifted Sternheimer equation from section 4.3 can be
  found in `compare_shifted_sternheimer`.
In each folder, you will find `.log` files of the SCF calculations, as well as the
response calculations, and `.json` files which store the data we need to
generate the plots.

# Usage
To perform the computations, first open the Julia shell with `julia --project`
from the location of this repository and then run
```
using Pkg
Pkg.instantiate()
```
to install the required dependencies.

To run the computations, two different cases arise:
- for silicon, open the Julia shell with `julia --project` from the `silicon/`
  folder and then run `include("silicon.jl")`;
- for the metals, first go, in your terminal, to the location of the system you
  want (e.g. `metals/Fe2MnAl`, `metals/Al40` or `compare_shifted_sternheimer/`)
  and then run for instance `mpiexecjl -n 8 julia Fe2MnAl.jl`.
  /!\ this take about an hour /!\

To generate all the plots from the paper, open the Julia shell with
`julia --project` from the location of this repository and then run
`include("all_plot.jl")`. Plots can be generated without running the
computations first as the `.json` files are already saved in the repo, and the
pdf files will be generated in the associated folders.

To run the adaptive algorithm from Section 6, open the Julia shell with
`julia --project` from the location of this repository and then run
`include("run_adaptive_algorithm.jl")`. Note thas this requires to first run
the computations for `Al40` et `Fe2MnAl` in the `metals` folder in order to
generate the `scfres.jld` files which store the data and are too large to be
stored in the repo.

# Contact
This is research code, not necessarily user-friendly, actively maintened or
extremely robust. If you have questions or encounter problems, get in touch!


