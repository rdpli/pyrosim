import cPickle
import random
import numpy as np
from replicators import Population, Individual

SEED = 1
POP_SIZE = 10
GENS = 10

DEVO = True

SECONDS = 60
DT = 0.05

SAVE_EVERY = 10
DIR = ''

random.seed(SEED)
np.random.seed(SEED)

# pop = Population(size=POP_SIZE, devo=DEVO, sec=SECONDS, dt=DT)
#
# for gen in range(GENS):
#     pop.create_children_through_mutation()
#     pop.add_random_inds(1)
#     pop.increment_ages()
#     pop.evaluate()
#     pop.reduce()
#     pop.print_non_dominated()
#     pop.gen += 1
#
#     if pop.gen % SAVE_EVERY == 0:
#         pop.save(DIR, SEED)

for run in range(13, 14):
    r = open(DIR + 'Rigid_Devo_1_Run_{}_Gen_10000.p'.format(run), 'r')
    pickle_dict = cPickle.load(r)

    robot_ids = []
    best_fit = 0
    champ_idx = 0
    for k, v in pickle_dict.items():
        robot_ids += [k]
        if v['fit'] > best_fit:
            champ_idx, best_fit = k, v['fit']
    print run, best_fit

# rand_idx = np.random.choice(robot_ids)
# champ_idx = rand_idx

bot = Individual(0, 1)
bot.weight_matrix = pickle_dict[champ_idx]['weights']
bot.devo_matrix = pickle_dict[champ_idx]['devo']

# bot.turn_off_brain()
# bot.turn_off_body(1.0)

bot.start_evaluation(seconds=SECONDS, dt=DT, blind=False)

