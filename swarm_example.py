import numpy as np
import random

class Particle:
    def __init__(self, n_stocks):
        self.position =  Particle.initial_swarm(n_stocks)
        self.pbest_position = self.position
        self.pbest_value = -10**3
        self.velocity = np.zeros(n_stocks)

    def move(self):
        proxy = self.position + self.velocity
        proxy = np.where(proxy>1, 1, proxy) # upper bound
        proxy = np.where(proxy<0, 0, proxy) # lower bound
        # print(proxy)
        self.position = proxy/np.sum(proxy)

    @staticmethod
    def initial_swarm(x):
        array = np.array([random.random() for i in range(x)])
        sum1 = np.sum(array)
        return array/sum1 # normalized weights

class Space:
    def __init__(self, n_particles, n_stocks, returns, vol, alpha, w, c1, c2):
        self.n_particles = n_particles
        self.particles = []
        self.gbest_value = -10**3
        self.alpha = alpha #sensitivity to sharpe
        self.n_stocks = n_stocks        
        self.gbest_position = Space.initial_swarm(n_stocks)
        self.returns = returns
        self.vol = vol #volatility matrix (variance)
        self.w = w
        self.c1 = c1
        self.c2 = c2

    @staticmethod
    def initial_swarm(x):
        array = np.array([random.random() for i in range(x)])
        sum1 = np.sum(array)
        return array/sum1 # normalized weights

    # fitness function for portfolio optimization
    def fitness(self, particle):
        return_ = np.dot(particle.position,np.transpose(self.returns))
        vol = np.dot(np.dot(particle.position,self.vol),np.transpose(particle.position))
        self.sharpe = return_/np.sqrt(vol) 

        # w\cdot covariance\:matrix\times w^{T}
        vols = np.dot(particle.position,self.vol)*particle.position
        
        # volatility\:dispersion=\sum (stock\:risk\:contibution_{i} -average\:risk)
        self.vol_dis = np.sum(abs(np.sqrt(vols)-np.mean(np.sqrt(vols))))
      
        # \frac{Sharpe\times\alpha+1 }{volaltility\:dispersion\times(1-\alpha)+1}
        result = (self.sharpe*self.alpha+1)/(self.vol_dis*(1-self.alpha)+1)

        return result

    def set_pbest(self):
        for particle in self.particles:
            fitness_candidate = self.fitness(particle)
            if (particle.pbest_value < fitness_candidate):
                particle.pbest_value = fitness_candidate
                particle.pbest_position = particle.position

    def set_gbest(self):
        for particle in self.particles:
            best_fitness_candidate = self.fitness(particle)
            if(self.gbest_value < best_fitness_candidate):
                self.gbest_value = best_fitness_candidate
                self.gbest_position = particle.position

    def move_particles(self):
        for particle in self.particles:
            new_velocity = (self.w*particle.velocity) + (self.c1*random.random()) * (particle.pbest_position - particle.position) + \
                            (random.random()*self.c2) * (self.gbest_position - particle.position)
            particle.velocity = new_velocity
            particle.move()