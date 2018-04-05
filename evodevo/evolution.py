import cPickle
import random
import numpy as np
from replicators import Population, Individual

SEED = 1
POP_SIZE = 30
GENS = 10000

DEVO = True

SECONDS = 60
DT = 0.05

SAVE_EVERY = 10
DIR = '/home/sam/Archive/skriegma/rigid_bodies/data'

random.seed(SEED)
np.random.seed(SEED)

# pop = Population(size=POP_SIZE, devo=DEVO, sec=SECONDS, dt=DT)
#
# for gen in range(GENS):
#     pop.create_children_through_mutation()
#     pop.add_random_inds(1)
#     pop.increment_ages()
#     pop.evaluate()
#     pop.update_hist()
#     pop.reduce()
#     pop.print_non_dominated()
#     pop.gen += 1
#
# pop.save(DIR, SEED)

for run in range(1, 2):
    print "loading run", run
    with open(DIR + '/Rigid_Devo_Run_{0}_Gen_{1}.p'.format(run, GENS), 'rb') as handle:
        pickle_dict = cPickle.load(handle)
    print "got it"

    robot_ids = []
    best_fit = 0
    champ_idx = 0
    for k, v in pickle_dict.items():
        robot_ids += [k]
        if v['fit'] > best_fit:
            champ_idx, best_fit = k, v['fit']
        # print run, best_fit

# rand_idx = np.random.choice(robot_ids)
# champ_idx = rand_idx

bot = Individual(0, 1)
bot.weight_matrix = pickle_dict[champ_idx]['weights']
bot.devo_matrix = pickle_dict[champ_idx]['devo']

print bot.devo_matrix

# bot.turn_off_brain()
# bot.turn_off_body(1.0)
print bot.calc_body_change(), bot.calc_control_change()

bot.start_evaluation(seconds=SECONDS, dt=DT, blind=False, fancy=True, pause=True)

