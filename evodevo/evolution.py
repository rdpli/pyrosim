import pickle
import random
import numpy as np
from replicators import Population

RUNS = 25
POP_SIZE = 100
GENS = 1000

SECONDS = 100
DT = 0.05


for run in range(RUNS):

    random.seed(run)
    np.random.seed(run)

    for devo in [False, True]:

        pop = Population(size=POP_SIZE, devo=devo, sec=SECONDS, dt=DT)

        for gen in range(GENS):
            pop.create_children_through_mutation()
            pop.add_random_inds(1)
            pop.increment_ages()
            pop.evaluate()
            pop.reduce()
            pop.print_non_dominated()
            pop.gen += 1

        f = open('~/scratch/Rigid_Devo/Rigid_Devo_{0}_Run_{1}.p'.format(int(devo), run), 'w')
        pickle.dump(pop, f)
        f.close()


# r = open('data/Dev_1_Run_1.p', 'r')
# final_pop = pickle.load(r)
#
# sorted_inds = sorted(final_pop.individuals_dict, key=lambda k: final_pop.individuals_dict[k].fitness)
# final_pop.individuals_dict[sorted_inds[-1]].start_evaluation(blind=False, eval_time=EVAL_TIME, pause=True)
#
# r.close()


