# run the adaptive algorithm from section 6

cd("metals/")
for dir in readdir()
    if isdir(dir) && !occursin("Nextra", String(dir))
        cd(dir)
        include("metals/$dir/run_adaptive_algo.jl")
        cd("../")
    end
end
cd("../")
