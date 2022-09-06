include("../common_plot.jl")

# plot histogram of convergence
plot_histo("Fe2MnAl")

# plot ratios of convergence
plot_ratios("Fe2MnAl", spin=true)

# plot data for k-points with index in ik_list
ik_list = [96, 236]
plot_cvg("Fe2MnAl", ik_list, spin=true)

# uncomment to plot convergence for every k-points
# [for Heusler compounds it will generate 280 pdf !]
#  all_plot("Fe2MnAl", spin=true)
