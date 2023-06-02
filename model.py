import itertools

import numpy as np

class Model:

	def __init__(self, n_loci, r_type='mult', **kwargs):
		self.n_loci = n_loci
		self.r_type = r_type
		
		self.S_genotypes = 2**self.n_loci
		self.G = np.array(list(itertools.product([0, 1], repeat=self.n_loci)))

		self.r_i = np.zeros(self.n_loci)
		self.c_i = np.zeros(self.n_loci)

		self.I_genotypes = 1

		self.b = 1
		self.mu = 0.2
		self.k = 0.001

		self.beta = 1

		for key, value in kwargs.items():
			setattr(self, key, value)

		self.rho = np.ones(self.n_loci - 1) * 0.5

		self.F = 1 - np.dot(self.G, self.c_i)
		self.B = self.transmission_matrix()
		#self.M = self.mating_matrix()

	def transmission_matrix(self):
		if self.r_type == 'mult':
			B = np.ones(self.S_genotypes) * self.beta

			for i, host in enumerate(self.G):
				for j in range(self.n_loci):
					if host[j] == 1:
						B[i] = B[i] * (1 - self.r_i[j])

		elif self.r_type == 'add':
			B = (1 - np.dot(self.G, self.r_i)) * self.beta
		
		return B
			
	def mating_matrix(self):
		paths = np.array(list(itertools.product([0, 1], repeat=self.n_loci)))
		p_paths = np.zeros(self.S_genotypes)

		for i, path in enumerate(paths):
			p_locus = np.zeros(self.n_loci)
			p_locus[0] = 0.5

			for j in range(self.n_loci - 1):
				if path[j] == path[j+1]:
					p_locus[j+1] = 1 - self.rho if np.isscalar(self.rho) else 1 - self.rho[j]
				else:
					p_locus[j+1] = self.rho if np.isscalar(self.rho) else self.rho[j]
			
			p_paths[i] = np.product(p_locus)

		M = np.zeros((self.S_genotypes, self.S_genotypes, self.S_genotypes))
		for i, parental in enumerate(self.G):
			for j, maternal in enumerate(self.G):
				offspring = np.zeros(paths.shape)
				offspring[paths==0] = np.tile(parental, (self.S_genotypes, 1))[paths==0]
				offspring[paths==1] = np.tile(maternal, (self.S_genotypes, 1))[paths==1]
				
				for k, progeny in enumerate(self.G):
					matches = np.product(offspring == progeny, axis=1, dtype=bool)
					M[i,j,k]= np.dot(matches, p_paths)
		
		M = M.reshape((self.S_genotypes**2, self.S_genotypes))
		return M