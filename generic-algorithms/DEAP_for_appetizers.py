# -*- coding: utf-8 -*-
"""
Created on Sun May 10 15:41:03 2020

@author: wyckl
"""


# The characters want to get exactly $15.05 worth of appetizers, as fast as possible 
# Appetizers include : 
    # Mixed fruit 2.15 
    # French fries 2.75 
    # Side Salad 3.35 
    # hot wings 3.55 
    # mozzarella sticks 4.20 
    # Smapler plate 5.80 
    
import random 
import sys
from operator import attrgetter
from collections import Counter 

try: 
    del Counter.__reduce__ # delete reduction function .. doesnt copy added attributes
except: pass     

import numpy as np
from deap import algorithms 
from deap import base 
from deap import creator 
from deap import tools 


class XKCD() : 
    
    def __init__(self): 
        
        # create the item dictionary , with id and (name, weight, value) tuple 
        
        self.ITEMS = {'French Fries': (2.75, 5), 
                      'Hot Wings': (3.55, 7), 
                      'Mixed Fruit': (2.15, 2), 
                      'Mozzarela Sticks': (4.2, 4), 
                      'Sampler Plate': (5.8, 10), 
                      'Side Salad': (3.35, 3)}
        self.ITEMS_NAMES = list(self.ITEMS.keys())
        
        self.individual_unit_size = 3 # initially create individuals that have 3 items 
        self.target_price = 15.05
        
        # load the simulator 
        print("Loading simulator ............................")
        self.optimizer()
 
        
    def optimizer(self): 
            # we are optimizing for three values : 
            # cost difference from target price (we want to spend as much as possible but not more than the budget)
            # Time to eat (max of time to each individual appetizers) we want to get out of here
            # amount of food (sum of counts of appetizers)
            
            # max cost difference, min time, maximize amount of food 
            creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0, 1.0)) # -1 means minimizing and 1 is maximizing 
            
            # individual withh eb a dict where the values are counts 
            creator.create("Individual", Counter, fitness=creator.FitnessMulti)
            
            toolbox = base.Toolbox()
            
            # individuals will be made up of countes associated with food items 
            toolbox.register("attr_item", random.choice, self.ITEMS_NAMES)
            toolbox.register("individual", tools.initRepeat, creator.Individual, 
                             toolbox.attr_item, self.individual_unit_size)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)   
            
            # register the functions, some with extra fixed arguments e.g. target price 
            toolbox.register('evaluate', self.evaluate, target_price=self.target_price)
            
            # severely penalize individuals that are over budgets and return fixed teribble fitness
            toolbox.decorate("evaluate", tools.DeltaPenality(lambda ind: self.evaluate(ind, self.target_price)[0] <= 0, 
                                                              (-sys.float_info.max, 
                                                               sys.float_info.max, 
                                                               -sys.float_info.max)))
            toolbox.register("mate", self.crossxCounter, indpb=0.5)
            toolbox.register("mutate", self.mutationCounter)
            toolbox.register("select", tools.selNSGA2)
            
            # Simulation parameters: 
            # Number of genrations
            
            # Number of generations
            ngen = 1000 
            # the number of individuals to select for the next generation (eliminate bad ones)
            mu = 500 
            # the number of children to produce at each generation 
            lambda_ = 200 
            # the probability that an offspring is produced by crossover 
            cxpb = 0.3 
            # the probability that an offspring is produced by mutation 
            mutpb = 0.3 
            
            # start population at 50 
            population = toolbox.population(n=mu)
            
            # keep track of the "hall of fame" 
            hof = tools.ParetoFront() # the best individual is the one that is equal or better  on all dimensions of the fitness function 
            # in this case (cost differnce and max time) so the hof is the one that is lowest on both 
            
            # compute some statistics as the simulation proceeds 
            price_stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
            time_stats  = tools.Statistics(key=lambda ind: ind.fitness.values[1])
            food_stats  = tools.Statistics(key=lambda ind: ind.fitness.values[2])
            
            stats = tools.MultiStatistics(price=price_stats, time=time_stats, food=food_stats)
            stats.register("avg", np.mean, axis=0)
        
            # run the simulation 
            print('Running simulation..............................')
            algorithms.eaMuPlusLambda(population, toolbox, mu, lambda_, cxpb, mutpb, ngen, stats, halloffame=hof)
            
          #  for i in hof: 
             #   print(i, i.fitness.values)
            
            print("Best:", hof[0], hof[0].fitness.values)
     
    def evaluate(self, individual, target_price): 
        
        # evaluates the fitness and returns the error in the price and the time taken by 
        # the order if the chef can cook everything in parallel 
        
        price = 0.0 
        times = [0]
        food  = 0 
        
        for item, number in individual.items() : 
            
            if number > 0 : 
                price += self.ITEMS[item][0] * number 
                times.append(self.ITEMS[item][1])
                food += number 
                
        return (price - target_price) , max(times) , food 
    
    def crossxCounter(self, ind1, ind2, indpb): 
        
        # Swaps the number of randomly-choses items between two individuals 
        
        for key in self.ITEMS.keys(): 
            if random.random() < indpb : 
                ind1[key] , ind2[key] = ind2[key], ind1[key]
                
        return ind1, ind2
    
    def mutationCounter(self, individual): 
        
        # adds or remove an item from an individual 
        if random.random() > 0.5:
            individual.update([random.choice(self.ITEMS_NAMES)]) #   make a counter go up 
        else:
            val = random.choice(self.ITEMS_NAMES)
            individual.subtract([val])
            if individual[val] < 0:
                del individual[val]  # take out the item with negative value
        return individual,

               
x = XKCD()