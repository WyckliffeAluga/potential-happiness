# -*- coding: utf-8 -*-
"""
Created on Sun May 10 15:05:13 2020

@author: wyckl
"""


class GA(max_generations): 
    
    def __init__(self, max_generations):
        
        self.max_generations  = max_generations
        
        
    def fitness_values(self): 
        
        # calculate fitness values and put the values along with the individuals, in a new weighted list 
        
        for generation in range(self.max_generations) :
            
            weighted_population  = [] 
            for individual in population : 
                fitness_val = self.fitness(individual)
                weighted_population.append((individual, fitness_val))
                
    def breeding(self): 
        
        individual1 = self.selection(self.fitness_values())
        individual2 = self.selection(self.fitness_values())