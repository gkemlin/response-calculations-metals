# run all plot files recursively

cd("silicon/")
include("silicon/plot.jl")

cd("../metals/")
for dir in readdir()
    if isdir(dir)
        cd(dir)
        include("metals/$dir/plot.jl")
        cd("../")
    end
end
cd("../")
