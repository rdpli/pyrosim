import numpy as np
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
        if devo:
            weight_matrix = np.random.rand(num_sensors+num_motors, num_sensors+num_motors, 2)
            self.devo_matrix = 2.0 * np.random.rand(8, 2) - 1.0
        else:
            weight_matrix = np.random.rand(num_sensors+num_motors, num_sensors+num_motors)
            self.devo_matrix = np.zeros((8, 2))

        self.weight_matrix = 2.0 * weight_matrix - 1.0

    def mutate(self, new_id, n=1):
        # neural net
        weight_change = np.random.normal(scale=np.abs(self.weight_matrix))
        new_weights = np.clip(self.weight_matrix + weight_change, -1, 1)
        mask = np.random.random(self.weight_matrix.shape) < n/float(weight_change.size)
        self.weight_matrix[mask] = new_weights[mask]

        # leg length
        if self.devo:
            devo_change = np.random.normal(scale=np.abs(self.devo_matrix))
            new_devo = np.clip(self.devo_matrix + devo_change, -1, 1)
            mask = np.random.random(self.devo_matrix.shape) < n/float(devo_change.size)
            self.devo_matrix[mask] = new_devo[mask]

        self.id = new_id
        self.already_evaluated = False

    def start_evaluation(self, seconds, dt):
        eval_time = int(seconds/dt)
        self.sim = pyrosim.Simulator(eval_time=eval_time, play_blind=True, dt=dt)
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

    def print_non_dominated(self):
        print self.gen, self.pareto_levels[0]

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
