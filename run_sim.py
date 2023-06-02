import numpy as np
from scipy.integrate import solve_ivp

from model import Model

def run_sim(model, af_S, af_I, t=(0,10000), init_hosts=400, init_inf=10, mut_rate=0.0001):

	def df(t, X):
		S = X[:model.S_genotypes]
		I = X[model.S_genotypes:]

		N = np.sum(S) + np.sum(I)
 
		genotype_freq = S / np.sum(S)
		pair_freq = np.outer(model.F*genotype_freq, genotype_freq).flatten()

		dS = np.sum(S)*np.dot(Mut, np.dot(pair_freq, model.M)) - \
			S*(model.k*N + model.mu + np.dot(model.B, I)/N)
		dI = I*(np.dot(model.B.T, S)/N - model.mu)

		X_out = np.append(dS, dI)

		return X_out

	#Assign host genotype ICs based on allele frequencies
	S_0 = np.ones(model.S_genotypes)		
	for i in range(model.n_loci):
		S_0[model.G[:,i] == 0] = S_0[model.G[:,i] == 0] * (1 - af_S[i])
		S_0[model.G[:,i] == 1] = S_0[model.G[:,i] == 1] * (af_S[i])
	
	#Assign infected ICs based on Avr frequency
	I_0 = af_I

	dist = np.zeros((model.S_genotypes, model.S_genotypes))
	
	for i, host_1 in enumerate(model.G):
		for j, host_2 in enumerate(model.G):
			dist[i, j] = np.linalg.norm(host_1 - host_2, 1)

	Mut = np.power(mut_rate, dist)*np.power(1-mut_rate, model.n_loci-dist)

	X_0 = np.append(S_0 * init_hosts, I_0 * init_inf)
	sol = solve_ivp(df, t, X_0, method='DOP853')

	S = sol.y[:model.S_genotypes, :]
	I = sol.y[model.S_genotypes:, :]

	return sol.t, S, I