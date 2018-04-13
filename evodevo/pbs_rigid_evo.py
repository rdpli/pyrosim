import random
import sys
import numpy as np
from replicators import Population

SEED = int(sys.argv[1])
POP_SIZE = 30
GENS = 3000

DEVO = False

SECONDS = 60
DT = 0.05

DIR = '/users/s/k/skriegma/scratch/rigid_bodies/data'

random.seed(SEED)
np.random.seed(SEED)

pop = Population(size=POP_SIZE, devo=DEVO, sec=SECONDS, dt=DT)

for gen in range(GENS):
    pop.create_children_through_mutation()
    pop.add_random_inds(1)
    pop.increment_ages()
    pop.evaluate()
    pop.update_hist()
    pop.reduce()
    pop.print_non_dominated()
    pop.gen += 1

pop.save(DIR, SEED)
