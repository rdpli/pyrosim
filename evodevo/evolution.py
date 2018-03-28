import pickle
import random
import numpy as np
from replicators import Population, Individual

RUNS = 1
POP_SIZE = 200
GENS = 1000

SECONDS = 60
DT = 0.05
DIR = ''


for run in range(1, RUNS+1):

    random.seed(run)
    np.random.seed(run)

    for devo in [True]:  # [False, True]

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
            results[key]['fit'] = ind.fitness

        f = open(DIR + 'Rigid_Devo_{0}_Run_{1}.p'.format(int(devo), run), 'w')
        pickle.dump(results, f)
        f.close()


r = open(DIR + 'Rigid_Devo_1_Run_1.p', 'r')
pickle_dict = pickle.load(r)

best_fit = 0
champ_idx = 0
for k, v in pickle_dict.items():
    if v['fit'] > best_fit:
        champ_idx, best_fit = k, v['fit']

bot = Individual(0, 1)
bot.weight_matrix = pickle_dict[champ_idx]['weights']

# bot.turn_off_brain()  # show only morphological change

bot.devo_matrix = pickle_dict[champ_idx]['devo']

bot.start_evaluation(seconds=SECONDS, dt=DT, blind=False)

