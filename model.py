import itertools
from itertools import combinations

import numpy as np
from scipy.integrate import solve_ivp

class PModel:
	'''
	The PModel class is used to define a discrete random loci model. All
	basic parameters for the model are contained here, as well as functions
	used to run a dynamical simulation with the model.
	'''

	def __init__(self, n_loci, r_locus, c_locus, **kwargs):
		self.n_loci = n_loci
		
		self.S_genotypes = 2**self.n_loci
		self.G = np.array(list(itertools.product([0, 1], repeat=self.n_loci)))

		self.r_i = r_locus
		self.c_i = c_locus

		self.b = 1				#Baseline birth rate
		self.mu = 0.2			#Death rate
		self.k = 0.001			#Coefficient of density-dependent growth
		self.mut = 0.00001		#Mutation rate
		self.beta = 0.005		#Baseline transmission rate

		for key, value in kwargs.items():
			setattr(self, key, value)

		self.F = self.b - np.dot(self.G, self.c_i)	#Initialize fecundity vector based on genotype costs
		self.B = self.transmission_matrix()			#Initialize transmission rate vector
		self.M = self.mutation_matrix()				#Initialize mutation matrix

	def transmission_matrix(self):
		#Calculate the transmission rate of each genotype by summing loci effects
		B = (1 - np.dot(self.G, self.r_i))
		
		return B
	
	def mutation_matrix(self):
		#Compute a matrix for how many alleles each genotype pair differs by
		dist_matrix = np.zeros((self.S_genotypes, self.S_genotypes))

		for i in range(self.S_genotypes):
			for j in range(self.S_genotypes):
				dist_matrix[i,j] = np.sum(np.abs(self.G[i,:] - self.G[j,:]))
		
		#For genotypes that differ by a single mutation, set the mutation proability to the mutation rate
		M = np.zeros(dist_matrix.shape)
		M[dist_matrix == 1] = self.mut
		M[dist_matrix == 0] = 1 - self.mut*self.n_loci

		return M

	def add_epistasis(self, order, p, sigma, mu=1):
		#Generate a list of all loci combinations of length order and take a random sample from that list
		pairs = np.array(list(combinations(range(self.n_loci), order)))
		sample = pairs[np.where(np.random.binomial(1, p, len(pairs)) == 1)]

		#Sample epistatic effects
		r_effects = np.random.normal(mu, sigma, len(sample))
		#c_effects = np.random.normal(1, sigma, len(sample))

		for i, ind in enumerate(sample):
			ind = list(ind)
			for gtp_ind, gtp in enumerate(self.G):
				if np.all(gtp[ind] == 1):
					self.B[gtp_ind] += np.sum(self.r_i[ind])
					self.B[gtp_ind] -= r_effects[i]*(np.sum(self.r_i[ind]))

	#Normalize the birthrates and transmission rates
	def normalize(self, b_min=0.2, b_max=1):
		self.F = (self.F - np.min(self.F)) / (np.max(self.F) - np.min(self.F))
		self.F *= (b_max - b_min)
		self.F += b_min

		self.F = self.F * b_max

		self.B = self.B - np.min(self.B)
		self.B = self.B / np.max(self.B)

		self.B = self.B * self.beta

	#Change the costs and benefits of each active allele
	def update_loci(self, r_locus, c_locus):
		self.r_i = r_locus
		self.c_i = c_locus

		self.B = self.transmission_matrix()
		self.F = self.b - np.dot(self.G, self.c_i)
	
	def run_sim(self, t=(0, 1000), n_gens=50, max_step=0.5):
		#Define initial conditions
		S_0 = np.zeros(self.S_genotypes)
		S_0[0] = 100
		I_0 = 10
		X_0 = np.append(S_0, I_0)
		
		X_t = np.zeros((self.S_genotypes, n_gens))	#Equilibrium host abundances
		I_t = np.zeros(n_gens)						#Equilibrium infected host abundances
		
		#Define the dynamical system	
		def df(t, X):
			S = X[:self.S_genotypes]
			I = X[-1]

			N = np.sum(S) + np.sum(I)
	
			dS = S*(self.F - self.k*N - self.mu - self.B*I)
			dI = I*(np.dot(self.B.T, S) - self.mu)

			X_out = np.append(dS, dI)

			return X_out

		#Burn in ecological dynamics before mutation
		X_0 = solve_ivp(df, (0, 100), X_0, max_step=max_step).y[:,-1]

		#Iteratively solve equations and introduce mutations
		for i in range(n_gens):
			sol = solve_ivp(df, t, X_0, max_step=max_step)
			
			X_t[:, i] = sol.y[:self.S_genotypes, -1]
			I_t[i] = sol.y[-1, -1]

			X_0 = np.append(np.dot(self.M, sol.y[:self.S_genotypes,-1]), sol.y[-1, -1])

		return X_t, I_t

	#Return the transmission and fecundity values for genotypes on the pareto front
	def pareto(self):
		res_pareto = []
		fec_pareto = []

		for i in range(len(self.B)):
			candidates = np.where(self.B <= self.B[i])

			if not np.any(self.F[candidates] > self.F[i]):
				res_pareto.append(self.B[i])
				fec_pareto.append(self.F[i])
		
		res_pareto = np.array(res_pareto)
		fec_pareto = np.array(fec_pareto)

		fec_pareto = fec_pareto[res_pareto.argsort()]
		res_pareto = np.sort(res_pareto)

		return res_pareto, fec_pareto

	#Return a polynomial approximation of the Pareto front
	def poly_approx(self, order=3, points=1000):
		res, fec = self.pareto()

		res_interp = np.linspace(np.min(res), np.max(res), points) 
		fec_pli = np.interp(res_interp, res, fec)

		coefs = np.polyfit(res_interp, fec_pli, order)
		fec_interp = np.poly1d(coefs)(res_interp)

		return res_interp, fec_interp, coefs
	
	'''
	def spline_approx(self, points=1000):
		res, fec = self.pareto()
		cs = CubicSpline(res, fec)

		res_interp = np.linspace(np.min(res), np.max(res), points)
		fec_interp = cs(res_interp) 

		return res_interp, fec_interp
	'''
		
	#Return a polynomial approximation of the Pareto front
	def linear_approx(self, points=1000):
		res, fec = self.pareto()

		res_interp = np.linspace(np.min(res), np.max(res), points) 
		fec_interp = np.interp(res_interp, res, fec)

		return res_interp, fec_interp

class ADModel:
	'''
	The ADModel class is used to define an adaptive dynamics simulation
	based on the same system of equations in the discrete random loci
	model. We use a numerical adaptive dynamics approach, where the
	cost function can be defined by defining the b and res vectors.
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

		#Set default parameters and resistance-cost curve
		self.res = np.linspace(0, 1, self.N_alleles)
		self.b = 0.5 + self.res**0.5

		#Set default parameters
		self.mu = 0.2		#Death rate
		self.k = 0.001		#Coefficient of density-dependent growth
		self.beta = 0.005	#Baseline transmission rate

		for key, value in kwargs.items():
			setattr(self, key, value)

	#Define the dynamical system	
	def df(self, t, X):
		S = X[:self.N_alleles] 
		I = X[-1]

		N = np.sum(S) + I
		dS = S*(self.b - self.mu - self.k*N - self.res*I)
		dI = I*(np.dot(self.res, S) - self.mu)
		
		X_out = np.append(dS, dI)

		return X_out

	#Run simulation
	def run_sim(self):
		#Define initial conditions
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

def PIP(res, fec):
	#Remove genotypes with zero transmission to prevent division by zero
	mask = np.where(res == 0)
	res = np.delete(res, mask)
	fec = np.delete(fec, mask)

	#Compute equilibrium susceptible prevalence
	def S_star(beta, mu=0.2):
		return mu / beta

	#Calculate equilibrium infected prevalence 
	def I_star(beta, b, mu=0.2, k=0.001):
		return (beta*(b - mu) - k*mu) / (beta*(k + beta))

	#Calculate pairwise invasion fitness
	def invasion_fitness(b_res, beta_res, b_mut, beta_mut, mu=0.2, k=0.001):
		return b_mut - mu - k*(S_star(beta_res) + I_star(beta_res, b_res)) - beta_mut*I_star(beta_res, b_res)

	PIP = np.zeros((len(res), len(res)))

	#Go through all genotype pairs and compute invasion fitness
	for i in range(PIP.shape[0]):
		for j in range(PIP.shape[1]):
			diff = invasion_fitness(fec[i], res[i], fec[j], res[j])

			if i == j:
				continue

			elif diff > 0:
				PIP[i, j] = 1

	return PIP