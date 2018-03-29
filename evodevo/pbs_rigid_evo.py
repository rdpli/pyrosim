import pickle
import random
import sys
import numpy as np
from replicators import Population

SEED = int(sys.argv[1])
POP_SIZE = 500
GENS = 2000

DEVO = False

SECONDS = 60
DT = 0.05
DIR = '/users/s/k/skriegma/scratch/rigid_bodies/'

random.seed(SEED)
np.random.seed(SEED)

pop = Population(size=POP_SIZE, devo=DEVO, sec=SECONDS, dt=DT)

for gen in range(GENS):
    pop.create_children_through_mutation()
    pop.add_random_inds(1)
    pop.increment_ages()
    pop.evaluate()
    pop.reduce()
    pop.print_non_dominated()
    pop.gen += 1

results = {key: {'weights': [], 'devo': [], 'age': 0, 'fit': 0} for key, ind in pop.individuals_dict.items()}
for key, ind in pop.individuals_dict.items():
    results[key]['weights'] = ind.weight_matrix
    results[key]['devo'] = ind.devo_matrix
    results[key]['age'] = ind.age
    results[key]['fit'] = ind.fitness

f = open(DIR + 'Rigid_Devo_{0}_Run_{1}.p'.format(int(DEVO), SEED), 'w')
pickle.dump(results, f)
f.close()

