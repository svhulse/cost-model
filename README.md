# The Emergence of Non-Linear Evolutionary Trade-offs and the Maintenance of Genetic Polymorphisms

This repository contains the code used to generate the data referenced in our manuscript. The code used to make each figure is also included.

The models themselves and the basic tools for them are contained in the `model.py` file. The code for running particular analyses and generating figures is contained across several jupyter notebooks. This repository can be used to generate all figures from the manuscript, except for figure 1, which was made in illustrator.

`model.py` contains two class structures. For documentation on how to use these classes, please refer to the jupyter notebook files.

* `PModel` is the primary model. With this class, you can define a discrete random loci model, add epistasis, and run evolutionary simulations.
* `ADModel` contains code for running an adaptive dynamics simulation with equivalent dynamics to the discrete random loci model.

This file also contains the method `PIP`, which is used for generating the pairwise invasion plots.

`gen_data.ipynb` runs a series of simulations for a given level of epistasis and generates summary statistics for each simulation. This includes the following:

* `Polymorphism`: Whether or not the numerical simulation ended in a polymorphism with two or more genotypes having abundance greater than 1.
* `PIPMin: Pareto`: The minimum number of genotypes which can invade across all resident genotypes for the Pareto genotypes.
* `PIPMin: Polynomial`: The minimum number of genotypes which can invade across all resident genotypes for the cubic polynomial interpolation.
* `PIPMin: Linear`: The minimum number of genotypes which can invade across all resident genotypes for the piecewise linear interpolation.
* `Shannon Diversity`: The Shannon diversity of the numerical simulation at equilibrium.
* `Nucleotide Diversity`: The nucleotide diversity of the numerical simulation at equilibrium.
* `Slope`: The slope of the Pareto front given by the quadratic coefficient.

The data for all simulations is then put into a pandas DataFrame and saved as a .csv file.

`data_analysis.ipynb` takes in the .csv files generated by `gen_data` and computes probabilities of polymorphisms for several scenarios.

All other jupyter notebooks are named by the figures they are used to generate.

`fig_2.ipynb` is used to generate genotype landscapes with various levels of epistasis.

`fig_3.ipynb` is used to run evolutionary simulations with the discrete random loci model as well as compare the equilibrium genetic diversity over batchs of simulations.

`fig_S1.ipynb` is used to make trade-off plots with varying numbers of loci.

`fig_S2.ipynb` is used to make trade-off plots with either synergistic or antagonistic epistasis.

`fig_S3.ipynb` is used to make trade-off plots with stronger and more likely epistatic effects.

`fig_S4.ipynb` is used to make trade-off plots where the costs and benefits of each allele are normally distributed.

`fig_S5.ipynb` is used to make figures demonstrating pairwise invasion plots for Pareto genotypes, piecewise linear trade-off functions and cubic polynomial trade-off functions.

Note that the time it takes to run a simulation can be greatly reduced by increasing the max_step parameter in the run_sim() function within PModel, but this will occationally result in numerical instability.