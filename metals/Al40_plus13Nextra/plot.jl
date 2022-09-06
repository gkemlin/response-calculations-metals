include("../common_plot.jl")

# plot ratios of convergence
plot_ratios("Al40"; ξ=2.2)

# plot data for k-points with index in ik_list
ik_list = [1, 2, 5]
plot_cvg("Al40", ik_list)

# uncomment to plot convergence for every k-points
# [for Heusler compounds it will generate 280 pdf !]
#  all_plot("Al40")
