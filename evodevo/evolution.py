import pickle
import random
import numpy as np
from replicators import Population

RUNS = 1
POP_SIZE = 5
GENS = 5

SECONDS = 100
DT = 0.05
DIR = ''


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

        results = {key: {'weights': [], 'devo': [], 'age': 0, 'fit': 0} for key, ind in pop.individuals_dict.items()}
        for key, ind in pop.individuals_dict.items():
            results[key]['weights'] = ind.weight_matrix
            results[key]['devo'] = ind.devo_matrix
            results[key]['age'] = ind.age
            results[key]['fitness'] = ind.fitness

        f = open(DIR + 'Rigid_Devo_{0}_Run_{1}.p'.format(int(devo), run), 'w')
        pickle.dump(results, f)
        f.close()


# r = open('data/Dev_1_Run_1.p', 'r')
# final_pop = pickle.load(r)
#
# sorted_inds = sorted(final_pop.individuals_dict, key=lambda k: final_pop.individuals_dict[k].fitness)
# final_pop.individuals_dict[sorted_inds[-1]].start_evaluation(blind=False, eval_time=EVAL_TIME, pause=True)
#
# r.close()


