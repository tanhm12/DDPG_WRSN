import random
import numpy as np
from collections import deque

import gym

MC_POS = np.array([0, 0])
MC_V = 5
MC_CHARGING_POWER = 5
WORST_REWARD = -8000
TIME_INTERVAL = 50


class Networks:
    def __init__(self, file: str):
        # self.time = 0
        self.number_of_nodes = None
        self.remaining_energy = None
        self.min_threshold = 540
        self.max_threshold = 8000
        self.min_E = None
        self.max_E = None
        self.ecr = None  # energy consumption rate
        self.remaining_time = None
        self.coords = None  # coordinate
        self.distance_matrix = None
        self._initialize(file)

    def _initialize(self, file: str):
        # the first place is for base station
        f = open(file)
        self.number_of_nodes = int(f.readline().split()[0])
        remaining_energy = [1]
        ecr = [1]
        coords = [np.array([int(x) for x in f.readline().split()])]
        distance_matrix = np.zeros((self.number_of_nodes+1, self.number_of_nodes+1))

        for i in range(1, self.number_of_nodes + 1):
            data = f.readline().split()
            coords.append(np.array([float(data[0]), float(data[1])]))
            ecr.append(float(data[2]))
            remaining_energy.append(float(data[3]))
            # print(i, pos, energy_consumption_rate, E_remain)

        self.remaining_energy = np.array(remaining_energy)
        self.min_E = np.array([self.min_threshold for _ in range(self.number_of_nodes + 1)])
        self.max_E = np.array([self.max_threshold for _ in range(self.number_of_nodes + 1)])
        self.ecr = np.array(ecr)
        self.coords = np.array(coords)
        self.remaining_time = np.divide(self.remaining_energy, self.ecr)

        fn = lambda pos1, pos2: np.sqrt(sum(t**2 for t in (pos1 - pos2)))
        for i in range(self.number_of_nodes+1):
            for j in range(i+1, self.number_of_nodes + 1):
                distance_matrix[i, j] = fn(self.coords[i], self.coords[j])
                distance_matrix[j, i] = distance_matrix[i, j]
        self.distance_matrix = distance_matrix

    def update(self, time: float, charged_node=None):
        self.remaining_time[1:] -= time
        self.remaining_energy = np.multiply(self.remaining_time, self.ecr)
        if charged_node is not None:
            self.remaining_time[charged_node] += time
        for i in range(1, len(self.remaining_time)):
            if self.remaining_energy[i] < self.min_E[i]:
                print(i, self.remaining_energy[i])
                # self.remaining_time[1:] += time
                # if charged_node is not None: self.remaining_time[charged_node] -= time
                return True
        return False


class MC:
    def __init__(self, pos_id: int, pos: np.ndarray, velocity: int, charging_power: float):
        self.pos_id = pos_id
        self.pos = pos
        self.v = velocity
        self.charging_power = charging_power

    def move_to(self, pos_id: int, net: Networks):
        time = net.distance_matrix[self.pos_id, pos_id] / self.v
        # print(self.pos_id, pos_id, time)
        self.pos_id = pos_id
        return time

    def charge(self, pos_id: int, ratio: float, net: Networks):
        delta_E = (net.max_E[pos_id] - net.remaining_energy[pos_id]) * ratio
        # print(net.remaining_energy[pos_id])
        net.remaining_energy[pos_id] += delta_E
        # print(net.remaining_energy[pos_id])
        net.remaining_time[pos_id] += delta_E / net.ecr[pos_id]
        return delta_E / self.charging_power


class Environment:
    """
    *state: n*1 : remaining time
    *action: 2*1 : node , charging ratio
    """
    def __init__(self, file: str, seed=0):
        self.seed = seed
        # random.seed(self.seed)
        self.net = Networks(file)
        self.memory = deque(maxlen=2000)
        self.mc = MC(0, MC_POS, MC_V, MC_CHARGING_POWER)
        self.state_shape = (self.net.number_of_nodes,)
        self.action_shape = (2,)
        self.state = None
        self.done = False
        self.action = [None, None]
        self.avg_remaining_time = None

    def reset(self):
        if len(self.memory) > 0:
            # print(len(self.memory))
            self.state = random.choice(self.memory)[0]
            # print(self.state)
        # the first reset, memory has nothing
        else:
            self.state = self.net.remaining_time[1:]
        self.mc = MC(0, MC_POS, MC_V, MC_CHARGING_POWER)
        self.action = [None, None]
        self.net.remaining_time = np.concatenate(([1], self.state))
        self.net.remaining_energy = np.multiply(self.net.remaining_time, self.net.ecr)
        self.avg_remaining_time = np.mean(self.state)
        return self.state

    def is_stuck_with(self, node: int, ratio: float):
        if node == self.action[0]:
            if ratio == 1. or ratio == 0.:
                return self.net.update(TIME_INTERVAL)

    def step(self, action):
        node = int(action[0] * self.net.number_of_nodes) + 1
        ratio = action[1]
        done = self.is_stuck_with(node, ratio)
        if done:
            return None, WORST_REWARD, TIME_INTERVAL, True, None
        moving_time = self.mc.move_to(node, self.net)
        done = self.net.update(moving_time)
        if done:
            return None, WORST_REWARD, moving_time, True, None
        charging_time = self.mc.charge(node, ratio, self.net)
        times = moving_time + charging_time
        done = self.net.update(charging_time)
        if done:
            return None, WORST_REWARD, times, True, None
        next_state = self.net.remaining_time[1:]
        avg_remaining_time = np.mean(next_state)
        reward = avg_remaining_time - self.avg_remaining_time
        self.avg_remaining_time = avg_remaining_time
        self.state = next_state
        self.action = [node, ratio]
        return self.state, reward, times, done, None

    def memorize(self, sample):
        self.memory.append(sample)

    def get_samples(self, batch_size):
        if len(self.memory) < batch_size:
            raise IndexError("Batch size is bigger than length of agent memory")
        else:
            return random.sample(self.memory, batch_size)