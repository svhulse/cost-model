# The Emergence of Non-Linear Evolutionary Trade-offs and the Maintenance of Genetic Polymorphisms

This repository contains the code used to generate the data referenced in our manuscript. The code used to make each figure is also included. A Preprint of the manuscript should be availible shortly.

## Code Organization

The models themselves and the basic tools for them are containd in the `model.py file`. The code for running particular analyses and generating figures is contained across several jupyter notebooks.

`model.py` contains two class structures. For documentation on how to use these classes, please refer to the jupyter notebook files.

* `PModel` is the primary model. With this class, you can define a discrete random loci model, add epistasis, and run evolutionary simulations on this model
* `ADModel` contains code for running an adaptive dynamics simulation with equivalent dynamics to the discrete random loci model.

`pareto.ipynb` is used to generate genotype landscapes with various levels of epistasis, and was used to generate Figure 2 in the manuscript.

`evol_model.ipynb` is used to run evolutionary simulations with the discrete random loci model as well as compare the equilibrium genetic diversity over batchs of simulations. It was used to generate Figure 3 in the manuscipt.

`adaptive_dynamics.ipynb` is used to first run an evolutionary model in the same was as `evol_model.ipynb`, but then use the resulting Pareto front as the basis of a trade-off function which is then fed into an adaptive dynamics model. It was used to genete Supplemental Figure 1.

`alt_loci_numbers` is used to make trade-off plots like in Figure 2, but with varying numbers of loci. It was used to generate Supplemental Figure 2.