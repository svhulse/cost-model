import numpy as np
import pandas as pd

from multiprocessing import Pool

from model import PModel, PIP

def shannon_div(X_t):
	eq_prop = X_t[:,-1] / np.sum(X_t[:,-1])
	eq_prop = eq_prop[eq_prop > 0]
	div = -np.sum(eq_prop*np.log(eq_prop))
	return div

def nucleotide_div(model, X_t):
	div = 0
	sum = np.sum(X_t[:, -1])
	
	for j in range(0, model.S_genotypes):
		for k in range(j + 1, model.S_genotypes):
			pi = np.sum(np.absolute(model.G[j] - model.G[k]))
			div += (X_t[j, -1]*X_t[k, -1]*pi) / sum**2
			
	return div

def get_slope(model, points=1000):
	res, fec = model.pareto()
	slope = np.polyfit(res, 1-fec, 2)[0]

	return slope

def batch(name, n_sims, epistasis, p1=0.3, p2=0.3, sigma1=0.3, sigma2=0.3, n_loci=9, n_gens=15, t=(0,1000), beta=0.005):
	#Initialize model
	model = PModel(n_loci, np.zeros(n_loci), np.zeros(n_loci), beta=beta)
	
	data = {'Loci': [], 
		 'Epistasis': [], 
		 'Polymorphism': [], 
		 'Shannon Diversity': [], 
		 'Nucleotide Diversity': [], 
		 'Slope': [],
		 'Invasion Genotypes: Pareto': [],
		 'Invasion Genotypes: Interp': []}

	threshold = 5

	for i in range(n_sims):
		print(i, end='\r')

		#Resample model parameters
		cost = np.random.exponential(0.1, n_loci)
		res = np.random.exponential(0.1, n_loci)
		model.update_loci(res, cost)

		if epistasis == 1:
			model.add_epistasis(2, p1, sigma1)
		elif epistasis == 2:
			model.add_epistasis(2, p1, sigma1)
			model.add_epistasis(3, p2, sigma2)

		model.normalize()
		
		data['Invasion Genotypes: Pareto'].append(np.min(np.sum(PIP(model), axis=1)))
		data['Invasion Genotypes: Interp'].append(np.min(np.sum(PIP(model, interp=True, order=3, points=1000), axis=1)))
		data['Loci'].append(n_loci)
		data['Epistasis'].append(epistasis)

		#Run dynamical simulation
		X_t, _ = model.run_sim(t, n_gens)

		data['Shannon Diversity'].append(shannon_div(X_t))
		data['Nucleotide Diversity'].append(nucleotide_div(model, X_t))

		data['Slope'].append(get_slope(model))

		#Check for polymorphism and update counters
		if np.sum(X_t[:, -1] > threshold) > 1:
			data['Polymorphism'].append(True)
		else:
			data['Polymorphism'].append(False)

	df = pd.DataFrame(data)
	df.to_csv(name + '.csv', index=False, mode='w')  
	
	return df

if __name__ == '__main__':
	batch('epistasis_2', 1000, 2)