# The Emergence of Non-Linear Evolutionary Trade-offs and the Maintenance of Genetic Polymorphisms

This repository contains the code used to generate the data referenced in our manuscript. The code used to make each figure is also included.

## Code Organization

The models themselves and the basic tools for them are containd in the `model.py` file. The code for running particular analyses and generating figures is contained across several jupyter notebooks. This repository can be used to generate all figures from the manuscript, except for figure 1, which was made in illustrator.

`model.py` contains two class structures. For documentation on how to use these classes, please refer to the jupyter notebook files.

* `PModel` is the primary model. With this class, you can define a discrete random loci model, add epistasis, and run evolutionary simulations on this model
* `ADModel` contains code for running an adaptive dynamics simulation with equivalent dynamics to the discrete random loci model.

All other jupyter notebooks are named by the figures they are used to generate.

`fig_2.ipynb` is used to generate genotype landscapes with various levels of epistasis.

`fig_3.ipynb` is used to run evolutionary simulations with the discrete random loci model as well as compare the equilibrium genetic diversity over batchs of simulations.

`fig_S1.ipynb` is used to make trade-off plots like in Figure 2, but with varying numbers of loci.

`fig_S2.ipynb` is used to make trade-off plots like in Figure 2, but with either synergistic or antagonistic epistasis.

`fig_S3.ipynb` is used to make trade-off plots like in Figure 2, but with stronger and more likely epistatic effects.

`fig_S4.ipynb` is used to first run an evolutionary model in the same was as `evol_model.ipynb`, but then use the resulting Pareto front as the basis of a trade-off function which is then fed into an adaptive dynamics model.

Note that the time it takes to run a simulation can be greatly reduced by increasing the max_step parameter in the run_sim() function within PModel, but this will occationally result in numerical instability.