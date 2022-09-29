using DFTK
using JLD2
using DataFrames
using LaTeXStrings
using Latexify

disable_threading()

include("../adaptive_choice_extra_bands.jl")

open("adaptive.log", "w") do io
    redirect_stdout(io) do
        DFTK.reset_timer!(DFTK.timer)
        adaptive_choice_Nextra("Fe2MnAl", 2.5)
        println(DFTK.timer)
    end
end

