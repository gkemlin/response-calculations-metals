using DFTK
using JLD2
using DataFrames
using LaTeXStrings
using Latexify

include("../adaptive_choice_extra_bands.jl")

open("adaptive.log", "w") do io
    redirect_stdout(io) do
        DFTK.reset_timer!(DFTK.timer)
        adaptive_choice_Nextra("Al40", 2.2)
        println(DFTK.timer)
    end
end

