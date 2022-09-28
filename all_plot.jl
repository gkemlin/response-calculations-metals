# run all plot files recursively

cd("silicon/")
include("silicon/plot.jl")

cd("../compare_shifted_sternheimer/")
include("compare_shifted_sternheimer/plot.jl")

cd("../metals/")
for dir in readdir()
    if isdir(dir)
        cd(dir)
        include("metals/$dir/plot.jl")
        cd("../")
    end
end
cd("../")
