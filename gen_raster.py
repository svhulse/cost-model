import numpy as np

from model import Model
from run_sim import run_sim

from matplotlib import pyplot as plt

n_loci = 10
n_sims = 100
n_steps = 100

def add_epistasis(model, std):
	epi_matrix = np.triu(np.ones((n_loci, n_loci)), 1)
	epi_matrix = epi_matrix * np.random.normal(1, std, (epi_matrix.shape))

	for i in range(n_loci):
		for j in range(n_loci):
			for k, gtp in enumerate(model.G):
				if gtp[i] == 1 and gtp[j] == 1:
					if i <= j:
						model.B[i] = model.B[i] * epi_matrix[i,j] 	

cost_grid = np.zeros((n_sims, n_steps))

for i in range(n_sims):
	cost = np.random.exponential(0.1, n_loci)
	res = np.random.exponential(0.1, n_loci)

	sim = Model(n_loci, r_type='mult', r_i = res, c_i = cost)

	res_grid = np.linspace(0, np.max(1-sim.B), n_steps)

	for j, val in enumerate(res_grid):
		cost_grid[i,j] = np.min((1-sim.F)[1-sim.B >= val])

cost_avg = np.average(cost_grid, axis=0)
cost_ste = np.std(cost_grid, axis=0) / np.sqrt(n_sims)

fig, ax = plt.subplots(ncols=2, nrows=2)

ax[0,0].scatter(1 - sim.B, 1 - sim.F)

ax[0,1].plot(res_grid, cost_avg, c='k')
ax[0,1].fill_between(res_grid, cost_avg-cost_ste, cost_avg+cost_ste)
ax[0,1].set_xlabel('resistance')
ax[0,1].set_ylabel('cost')

cost_grid = np.zeros((n_sims, n_steps))

for i in range(n_sims):
	cost = np.random.exponential(0.1, n_loci)
	res = np.random.exponential(0.1, n_loci)

	sim = Model(n_loci, r_type='mult', r_i = res, c_i = cost)
	add_epistasis(sim, 0.1)

	res_grid = np.linspace(0, np.max(1-sim.B), n_steps)

	for j, val in enumerate(res_grid):
		cost_grid[i,j] = np.min((1-sim.F)[1-sim.B >= val])


cost_avg = np.average(cost_grid, axis=0)
cost_ste = np.std(cost_grid, axis=0) / np.sqrt(n_sims)

ax[1,0].scatter(1 - sim.B, 1 - sim.F)

ax[1,1].plot(res_grid, cost_avg, c='k')
ax[1,1].fill_between(res_grid, cost_avg-cost_ste, cost_avg+cost_ste)
ax[1,1].set_xlabel('resistance')
ax[1,1].set_ylabel('cost')