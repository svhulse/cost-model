import itertools
from itertools import combinations, chain

import random
import numpy as np
from scipy.integrate import solve_ivp

class PModel:

	def __init__(self, n_loci, r_locus, c_locus, **kwargs):
		self.n_loci = n_loci
		
		self.S_genotypes = 2**self.n_loci
		self.G = np.array(list(itertools.product([0, 1], repeat=self.n_loci)))

		self.r_i = r_locus
		self.c_i = c_locus

		self.b = 1
		self.mu = 0.2
		self.k = 0.001
		self.mut = 0.00001
		self.beta = 0.01

		for key, value in kwargs.items():
			setattr(self, key, value)

		self.F = self.b - np.dot(self.G, self.c_i)
		self.B = self.transmission_matrix()
		self.M = self.mutation_matrix()

	def transmission_matrix(self):
		B = (1 - np.dot(self.G, self.r_i))
		
		return B
	
	def mutation_matrix(self):
		dist_matrix = np.zeros((self.S_genotypes, self.S_genotypes))

		for i in range(self.S_genotypes):
			for j in range(self.S_genotypes):
				dist_matrix[i,j] = np.sum(np.abs(self.G[i,:] - self.G[j,:]))
		
		M = np.zeros(dist_matrix.shape)
		M[dist_matrix == 1] = self.mut
		M[dist_matrix == 0] = 1 - self.mut*self.S_genotypes

		return M

	def add_epistasis(self, order, p, sigma):
		pairs = np.array(list(combinations(range(self.n_loci), order)))
		sample = pairs[np.where(np.random.binomial(1, p, len(pairs)) == 1)]

		r_effects = np.random.normal(1, sigma, len(sample))
		#c_effects = np.random.normal(1, sigma, len(sample))

		for i, ind in enumerate(sample):
			ind = list(ind)
			for gtp_ind, gtp in enumerate(self.G):
				if np.all(gtp[ind] == 1):
					self.B[gtp_ind] += np.sum(self.r_i[ind])
					self.B[gtp_ind] -= r_effects[i]*(np.sum(self.r_i[ind]))

	def normalize(self, b_min=0.2):
		self.F = self.F - np.min(self.F) + b_min*np.max(self.F)
		self.F = self.F / np.max(self.F)

		self.B = self.B - np.min(self.B)
		self.B = self.B / np.max(self.B)

	def update_loci(self, r_locus, c_locus):
		self.r_i = r_locus
		self.c_i = c_locus

		self.B = self.transmission_matrix()
		self.F = self.b - np.dot(self.G, self.c_i)
	
	def run_sim(self, t=(0, 1000), n_gens=50):
		#Assign ICs based on allele frequencies
		S_0 = np.zeros(self.S_genotypes)

		S_0[0] = 50		
		I_0 = 1
		X_0 = np.append(S_0, I_0)
		
		X_t = np.zeros((self.S_genotypes, n_gens))
		I_t = np.zeros(n_gens)
		
		def df(t, X):
			S = X[:self.S_genotypes]
			I = X[self.S_genotypes:]

			N = np.sum(S) + np.sum(I)
	
			dS = S*(self.F - self.k*N - self.mu - self.B*I)
			dI = I*(np.dot(self.B.T, S) - self.mu)

			X_out = np.append(dS, dI)

			return X_out

		#Burn in ecological dynamics before mutation
		X_0 = solve_ivp(df, (0, 100), X_0).y[:,-1]

		for i in range(n_gens):
			sol = solve_ivp(df, t, X_0)
			
			X_t[:, i] = sol.y[:self.S_genotypes, -1]
			I_t[i] = sol.y[self.S_genotypes:, -1]

			X_0 = np.append(np.dot(self.M, sol.y[:self.S_genotypes,-1]), sol.y[self.S_genotypes:, -1])

		return X_t, I_t

class ADModel:
	'''
	The Model class is used to define a simulation for the QR host-pathogen
	model. It allows for both density-dependent and frequency-dependent disease
	transmission. Parameters can also be changed between multiple runs by 
	using kwargs in the run_sim method.
	'''

	def __init__(self, **kwargs):
		self.N_alleles = 100 #number of alleles
		self.N_iter = 50 #number of evolutionary time steps
		mut = 0.05 #mutation rate

		#Define the mutation matrix
		self._M = np.diag(np.full(self.N_alleles, 1 - mut))
		self._M = self._M + np.diag(np.ones(self.N_alleles - 1)*mut/2, 1)
		self._M = self._M + np.diag(np.ones(self.N_alleles - 1)*mut/2, -1)
		self._M[0,1] = mut
		self._M[self.N_alleles - 1, self.N_alleles - 2] = mut

		#Set parameters and resistance-cost curve
		self.res = np.linspace(0, 1, self.N_alleles)
		self.b = 0.5 + self.res**0.5

		#Set default parameters
		self.mu = 0.2
		self.gamma = 0.001
		self.beta = 0.005

		#Get kwargs from model initialization and modify parameter values, if
		#h is changed, then _t and N_t must also be changed to compensate
		for key, value in kwargs.items():
			setattr(self, key, value)
		
	def df(self, t, X):
		S = X[:self.N_alleles] 
		I = X[-1]

		N = np.sum(S) + I
		dS = S*(self.b - self.mu - self.gamma*N - self.res*I)
		dI = I*(np.dot(self.res, S) - self.mu)
		
		X_out = np.append(dS, dI)

		return X_out

	#Run simulation
	def run_sim(self):
		#Set initial conditions
		S_0 = np.zeros(self.N_alleles)
		I_0 = 10
		S_0[49] = 100
		X_0 = np.append(S_0, I_0)

		S_eq = np.zeros((self.N_alleles, self.N_iter))
		I_eq = np.zeros(self.N_iter)

		t = (0, 10000)
		zero_threshold = 0.01 #Threshold to set abundance values to zero

		for i in range(self.N_iter):
			sol = solve_ivp(self.df, t, X_0)
			
			S_eq[:, i] = sol.y[:self.N_alleles, -1]
			I_eq[i] = sol.y[-1, -1]

			#Set any population below threshold to 0
			for j in range(self.N_alleles):
				if S_eq[j, i] < zero_threshold:
					S_eq[j, i] = 0

			#Assign the values at the end of the ecological simulation to the 
			#first value so the simulation can be re-run
			X_0 = np.append(np.dot(self._M, S_eq[:, i]), I_eq[i])

		return (S_eq, I_eq)