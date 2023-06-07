import numpy as np

from model import Model
from run_sim import run_sim

from matplotlib import pyplot as plt

def add_epistasis(model, std):
	epi_matrix = np.triu(np.ones((model.n_loci, model.n_loci)), 1)
	epi_matrix = epi_matrix * np.random.normal(1, std, (epi_matrix.shape))

	for i in range(model.n_loci):
		for j in range(i+1, model.n_loci):
			for k, gtp in enumerate(model.G):
				if gtp[i] == 1 and gtp[j] == 1:
					model.B[k] = model.B[k] * epi_matrix[i,j] 	

def get_gtps(n_loci):
		cost = np.random.exponential(0.1, n_loci)
		res = np.random.exponential(0.1, n_loci)

		sim = Model(n_loci, r_type='mult', r_i = res, c_i = cost)
		sim.F = sim.F - np.min(sim.F)
		sim.F = sim.F / np.max(sim.F)

		sim.B = sim.B - np.min(sim.B)
		sim.B = sim.B / np.max(sim.B)

		return sim

def sim_array(n_loci, n_sims, n_steps=100, epi=False):
	cost_grid = np.zeros((n_sims, n_steps))

	for i in range(n_sims):
		sim = get_gtps(n_loci)

		if epi:
			add_epistasis(sim, 0.4)
			print('tha fuck')

		res_grid = np.linspace(0, np.max(1-sim.B), n_steps)

		for j, val in enumerate(res_grid):
			cost_grid[i,j] = np.min((1-sim.F)[1-sim.B >= val])

	cost_avg = np.average(cost_grid, axis=0)
	cost_ste = np.std(cost_grid, axis=0) / np.sqrt(n_sims)

	return cost_avg, cost_ste

n_loci = 9
costs, err = sim_array(n_loci, 100)
#costs_epi, err_epi = sim_array(n_loci, 100, epi=True)

res = np.linspace(0, 1, 100)

fig, ax = plt.subplots(ncols=2, nrows=2)

cost_dist = np.random.exponential(0.1, n_loci)
res_dist = np.random.exponential(0.1, n_loci)

sim = get_gtps(n_loci)

ax[0,0].scatter(1 - sim.B, 1 - sim.F)

ax[0,1].plot(res, costs, c='k')
ax[0,1].fill_between(res, costs-err, costs+err)
ax[0,1].set_xlabel('resistance')
ax[0,1].set_ylabel('cost')


#ax[1,0].scatter(1 - sim.B, 1 - sim.F)

ax[1,1].plot(res, costs_epi, c='k')
ax[1,1].fill_between(res, costs_epi-err_epi, costs_epi+err_epi)
ax[1,1].set_xlabel('resistance')
ax[1,1].set_ylabel('cost')
