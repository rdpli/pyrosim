import numpy as np
import cPickle
from copy import deepcopy
import pyrosim
from vehicles import send_to_simulator


class Individual(object):
    def __init__(self, idx, devo):

        self.id = idx
        self.devo = devo
        self.fitness_sensor_idx = 0
        self.fitness = 0
        self.age = 0
        self.dominated_by = []
        self.pareto_level = 0
        self.already_evaluated = False
        self.sim = []

        num_sensors = 5
        num_motors = 8

        weight_matrix = np.random.rand(num_sensors+num_motors, num_sensors+num_motors, 2)
        self.devo_matrix = 2.0 * np.random.rand(8, 2) - 1.0
        self.weight_matrix = 2.0 * weight_matrix - 1.0

        if not devo:
            self.remove_devo()

    def remove_devo(self):
        self.weight_matrix = np.stack([self.weight_matrix[:, :, 0], self.weight_matrix[:, :, 0]], axis=2)
        self.devo_matrix = np.stack([self.devo_matrix[:, 0], self.devo_matrix[:, 0]], axis=1)

    def turn_off_brain(self):
        self.weight_matrix = np.zeros_like(self.weight_matrix)

    def turn_off_body(self, l):
        self.devo_matrix = np.ones_like(self.devo_matrix)*l

    def calc_body_change(self):
        change = np.abs(self.devo_matrix[:, 0] - self.devo_matrix[:, 1])
        change /= float(2.0*len(change))
        return np.sum(change)

    def calc_control_change(self):
        change = np.abs(self.weight_matrix[:, :, 0] - self.weight_matrix[:, :, 1])
        change /= float(2.0*np.product(change.shape))
        return np.sum(change)

    def mutate(self, new_id, p=0.1):

        # if self.devo:
        #     n *= 2  # same proportion of genes mutated in evo and evo-devo

        # neural net
        weight_change = np.random.normal(scale=np.abs(self.weight_matrix))
        new_weights = np.clip(self.weight_matrix + weight_change, -1, 1)
        # mask = np.random.random(self.weight_matrix.shape) < n/float(weight_change.size)
        mask = np.random.random(self.weight_matrix.shape) < p
        self.weight_matrix[mask] = new_weights[mask]

        # leg length
        devo_change = np.random.normal(scale=np.abs(self.devo_matrix))
        new_devo = np.clip(self.devo_matrix + devo_change, -1, 1)
        # mask = np.random.random(self.devo_matrix.shape) < n/float(devo_change.size)
        mask = np.random.random(self.devo_matrix.shape) < p
        self.devo_matrix[mask] = new_devo[mask]

        if not self.devo:
            self.remove_devo()  # maintain a single leg length throughout development

        self.id = new_id
        self.already_evaluated = False

    def start_evaluation(self, seconds, dt, blind=True, fancy=False, pause=False):
        eval_time = int(seconds/dt)
        self.sim = pyrosim.Simulator(eval_time=eval_time, play_blind=blind, dt=dt,
                                     use_textures=fancy, play_paused=pause)
        layout = send_to_simulator(self.sim, weight_matrix=self.weight_matrix, devo_matrix=self.devo_matrix)
        self.sim.start()
        self.fitness_sensor_idx = layout['light_sensor']

    def compute_fitness(self):
        self.sim.wait_to_finish()
        dist = self.sim.get_sensor_data(self.fitness_sensor_idx)
        self.fitness = dist[-1]
        self.already_evaluated = True

    def dominates(self, other):

        if self.fitness > other.fitness and self.age <= other.age:
            return True

        elif self.fitness == other.fitness and self.age < other.age:
            return True

        elif self.fitness == other.fitness and self.age == other.age and self.id < other.id:
            return True

        else:
            return False


class Population(object):
    def __init__(self, size, devo, sec, dt):
        self.size = size
        self.devo = devo
        self.sec = sec
        self.dt = dt
        self.gen = 0
        self.individuals_dict = {}
        self.max_id = 0
        self.non_dominated_size = 0
        self.pareto_levels = {}
        self.add_random_inds(size)
        self.evaluate()
        self.hist = {key: {'weights': [], 'devo': [], 'age': 0, 'fit': 0} for key, ind in self.individuals_dict.items()}
        if devo:
            name = 'Devo'
        else:
            name = 'Evo'
        self.name = name

    def print_non_dominated(self):
        print self.gen, self.pareto_levels[0]

    def update_hist(self):
        for key, ind in self.individuals_dict.items():
            if key not in self.hist:
                self.hist[key] = {}
            self.hist[key]['weights'] = ind.weight_matrix
            self.hist[key]['devo'] = ind.devo_matrix
            self.hist[key]['age'] = ind.age
            self.hist[key]['fit'] = ind.fitness

    def save(self, dir, seed):
        f = open(dir + '/Rigid_{0}_Run_{1}_Gen_{2}.p'.format(self.name, seed, self.gen), 'w')
        cPickle.dump(self.hist, f)
        f.close()

    def evaluate(self):
        for key, ind in self.individuals_dict.items():
            if not ind.already_evaluated:
                ind.start_evaluation(self.sec, self.dt)

        for key, ind in self.individuals_dict.items():
            if not ind.already_evaluated:
                ind.compute_fitness()

    def create_children_through_mutation(self, fill_pop_from_non_dom=True):
        if fill_pop_from_non_dom:
            while len(self.individuals_dict) < self.size:
                for key, ind in self.individuals_dict.items():
                    child = deepcopy(ind)
                    child.mutate(self.max_id)
                    child.already_evaluated = False
                    self.individuals_dict[self.max_id] = child
                    self.max_id += 1

        else:
            for key, ind in self.individuals_dict.items():
                child = deepcopy(ind)
                child.mutate(self.max_id)
                child.already_evaluated = False
                self.individuals_dict[self.max_id] = child
                self.max_id += 1

    def increment_ages(self):
        for key, ind in self.individuals_dict.items():
            ind.age += 1

    def add_random_inds(self, num_random=1):
        for _ in range(num_random):
            self.individuals_dict[self.max_id] = Individual(self.max_id, self.devo)
            self.max_id += 1

    def update_dominance(self):
        for key, ind in self.individuals_dict.items():
            ind.dominated_by = []

        for key1, ind1 in self.individuals_dict.items():
            for key2, ind2 in self.individuals_dict.items():
                if key1 != key2:
                    if self.individuals_dict[key1].dominates(self.individuals_dict[key2]):
                        self.individuals_dict[key2].dominated_by += [key1]

        self.non_dominated_size = 0
        self.pareto_levels = {}
        for key, ind in self.individuals_dict.items():
            ind.pareto_level = len(ind.dominated_by)
            if ind.pareto_level in self.pareto_levels:
                self.pareto_levels[ind.pareto_level] += [(ind.id, ind.fitness, ind.age)]
            else:
                self.pareto_levels[ind.pareto_level] = [(ind.id, ind.fitness, ind.age)]
            if ind.pareto_level == 0:
                self.non_dominated_size += 1

    def reduce(self, keep_non_dom_only=True, pairwise=False):
        self.update_dominance()

        if keep_non_dom_only:  # completely reduce to non-dominated front (most pressure, least diversity)

            children = {}
            for idx, fit, age in self.pareto_levels[0]:
                children[idx] = self.individuals_dict[idx]
            self.individuals_dict = children

        elif pairwise:  # reduce by calculating pairwise dominance (least pressure, most diversity)

            while len(self.individuals_dict) > self.size and len(self.individuals_dict) > self.non_dominated_size:
                current_ids = [idx for idx in self.individuals_dict]
                np.random.shuffle(current_ids)
                inds_to_remove = []
                for idx in range(1, len(self.individuals_dict)):
                    this_id = current_ids[idx]
                    previous_id = current_ids[idx-1]
                    if self.individuals_dict[previous_id].dominates(self.individuals_dict[this_id]):
                        inds_to_remove += [this_id]
                for key in inds_to_remove:
                    del self.individuals_dict[key]

        else:  # add by pareto level until full

            children = {}
            for level in sorted(self.pareto_levels):
                sorted_level = sorted(self.pareto_levels[level], key=lambda x: x[1], reverse=True)
                for idx, fit, age, compression in sorted_level:
                    if len(children) < self.size:
                        children[idx] = self.individuals_dict[idx]
            self.individuals_dict = children

