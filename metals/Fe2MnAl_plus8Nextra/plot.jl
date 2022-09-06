include("../common_plot.jl")

# plot ratios of convergence
plot_ratios("Fe2MnAl", spin=true; ξ=2.5)

# plot data for k-points with index in ik_list
ik_list = [96, 236, 72, 212]
plot_cvg("Fe2MnAl", ik_list, spin=true)

# uncomment to plot convergence for every k-points
# [for Heusler compounds it will generate 280 pdf !]
#  all_plot("Fe2MnAl", spin=true)
