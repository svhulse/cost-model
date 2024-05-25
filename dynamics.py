import numpy as np
from scipy.integrate import solve_ivp

from model import PModel

def run_sim(model, t=(0,1), n_gens=50):

	def df(t, X):
		S = X[:model.S_genotypes]
		I = X[model.S_genotypes:]

		N = np.sum(S) + np.sum(I)
 
		dS = S*(model.F - model.k*N - model.mu - model.B*I)
		dI = I*(np.dot(model.B.T, S) - model.mu)

		X_out = np.append(dS, dI)

		return X_out

	#Assign ICs based on allele frequencies
	S_0 = np.zeros(model.S_genotypes)

	S_0[0] = 50		
	I_0 = 1
	X_0 = np.append(S_0, I_0)

	X_t = np.zeros((model.S_genotypes, n_gens))
	I_t = np.zeros(n_gens)

	#Burn in ecological dynamics before mutation
	X_0 = solve_ivp(df, (0, 100), X_0).y[:,-1]

	for i in range(n_gens):
		sol = solve_ivp(df, t, X_0)
		
		X_t[:, i] = sol.y[:model.S_genotypes, -1]
		I_t[i] = sol.y[model.S_genotypes:, -1]

		X_0 = np.append(np.dot(model.M, sol.y[:model.S_genotypes,-1]), sol.y[model.S_genotypes:, -1])

	return X_t, I_t