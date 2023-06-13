import numpy as np
import matplotlib.pyplot as plt

class Squaresimulation:
    def __init__(self, n):
        self.n = n
        self.lattice = self.create_lattice()
        
    def create_lattice(self):
        lattice = np.random.choice([-1, 1], size=(self.n, self.n))
        return lattice
    
    def compute_magnetization(self):
        N  = self.lattice.shape[0] * self.lattice.shape[1]
        magnetization = np.sum(self.lattice) / N
        return magnetization
    
    def compute_neighbor_sum(self, row, col):
        n = self.lattice.shape[0]
        neighbor_sum = 0
        
        if row > 0:
            neighbor_sum += self.lattice[row-1, col]
        
        if row < n-1:
            neighbor_sum += self.lattice[row+1, col]
        
        if col > 0:
            neighbor_sum += self.lattice[row, col-1]
        
        if col < n-1:
            neighbor_sum += self.lattice[row, col+1]
        
        return neighbor_sum
    
    def compute_boltzmann_factor(self, row, col, beta, J, h):
        neighbor_sum = self.compute_neighbor_sum(row, col)
        p = np.exp(beta * (J * neighbor_sum + h))
        m = np.exp(-beta * (J * neighbor_sum + h))
        return p, m
    
    def compute_conditional_probability(self, row, col, beta, J, h):
        p, m = self.compute_boltzmann_factor(row, col, beta, J, h)
        p, m = p / (p + m), m / (p + m)
        s = np.random.choice([1, -1], p=[p, m])
        return s, p, m
    
    def update_lattice(self, s, row, col):
        self.lattice[row, col] = s
    
    def one_step(self, beta, J, h):
        n1 = self.lattice.shape[0]
        n2 = self.lattice.shape[1]
        
        for row in range(n1):
            for col in range(n2):
                s, p, m = self.compute_conditional_probability(row, col, beta, J, h)
                self.update_lattice(s, row, col)
    
    def compute_magnetization_transition(self, beta, J, h, n_steps):
        magnetization_transition = []
        
        for i in range(n_steps):
            self.one_step(beta, J, h)
            m = self.compute_magnetization()
            magnetization_transition.append(m)
        
        return magnetization_transition
