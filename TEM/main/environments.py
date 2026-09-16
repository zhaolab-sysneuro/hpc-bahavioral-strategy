#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: James Whittington
"""

import numpy as np
import copy as cp
import itertools
import random

class Environment:
    def __init__(self, params, width, height, n_states, test=False):
        super(Environment, self).__init__()
        self.par = params
        self.width = width
        self.height = height
        self.n_actions = self.par.env.n_actions
        self.rels = self.par.env.rels
        self.walk_len = None
        self.reward_value = 1.0
        self.reward_pos_training = []
        self.start_state, self.adj, self.tran, self.states_mat = None, None, None, None

        if n_states > self.par.max_states:
            raise ValueError(
                ('Too many states in your world. {} is bigger than {}. Adjust by decreasing environment size, or' +
                 'increasing params.max_states').format(n_states, self.par.max_states))


class Zhao2025(Environment):
    '''
    [S,A,B,C,E]
    [S,A,C,B,E]
    [S,B,A,C,E]
    [S,B,C,A,E]
    [S,C,A,B,E]
    [S,C,B,A,E]
    '''
    def __init__(self, params, width, test):
        self.n_states = width * 6
        super().__init__(params, width, width, self.n_states)

        # reward locations
        self.reward_A_pos = [1,6,12,18,22,28]
        self.reward_B_pos = [2,8,11,16,23,27]
        self.reward_C_pos = [3,7,13,17,21,26] 
        self.start_pos = [0,5,10,15,20,25]
        self.end_pos = [4,9,14,19,24,29]
        self.reward_pos_training = [] # self.reward_A_pos + self.reward_C_pos
        self.reward_value = self.n_states
        self.test = test
        self.start_state = np.random.choice(6) * 5

    def world(self):
        width = self.width
        n_states =width * 6
        adj = np.zeros((n_states, n_states))

        # go round track twice
        for i in range(n_states):
            if i < n_states - 1:
                adj[i, i + 1] = 1
        s_start = [width * x for x in range(6)]
        s_end = [width * x + width - 1 for x in range(6)]
        for state in s_end:
            adj[state, s_start] = 1
        
        tran = np.zeros((n_states, n_states))
        for i in range(n_states):
            if sum(adj[i]) > 0:
                tran[i] = adj[i] / sum(adj[i])

        self.adj, self.tran = adj, tran

    def relation(self, s1, s2):
        width = self.width
        n_states = width * 6
        pos_lap_1 = s1 % width
        pos_lap_2 = s2 % width

        if s1 > n_states or s2 > n_states:
            raise ValueError('impossible state index - too high')
        if pos_lap_2 - pos_lap_1 == 0:
            rel_type = 'stay still'
        elif (pos_lap_2 - pos_lap_1) == 1:
            rel_type = 'run'
        elif pos_lap_1 == width - 1 and pos_lap_2 == 0:
            rel_type = 'run'
        else:
            raise ValueError('impossible action')

        rel_index = self.rels.index(rel_type)

        return rel_index, rel_type

    def state_data(self):
        '''
        [S,A,B,C,E]
        [S,A,C,B,E]
        [S,B,A,C,E]
        [S,B,C,A,E]
        [S,C,A,B,E]
        [S,C,B,A,E]
        '''
        states_vec = np.zeros(self.n_states)
        choice_size = self.par.s_size

        # choose reward sense
        reward_choices = np.random.choice(choice_size, size=4, replace=False)
        reward_A_sense = reward_choices[1]
        reward_B_sense = reward_choices[2]
        reward_C_sense = reward_choices[1]
        start_sense = reward_choices[0]
        end_sense = reward_choices[3]

        # make particular position special in track
        for r_p in self.reward_A_pos:
            states_vec[r_p] = reward_A_sense
        for r_p in self.reward_B_pos:
            states_vec[r_p] = reward_B_sense
        for r_p in self.reward_C_pos:
            states_vec[r_p] = reward_C_sense    
        for r_p in self.start_pos:
            states_vec[r_p] = start_sense
        for r_p in self.end_pos:
            states_vec[r_p] = end_sense                
        self.states_mat = states_vec.astype(int)

    def walk(self):
        time_steps = self.walk_len
        position = np.zeros(time_steps, dtype=np.int16)
        direc = np.zeros((self.n_actions, time_steps))

        position[0] = int(self.start_state)
        # choose random action to have gotten to start-state - doesn't get used as g_prior is for first state
        direc[0, 0] = 1
        for i in range(time_steps - 1):
            available = np.where(self.tran[int(position[i]), :] > 0)[0].astype(int)
            p = self.tran[int(position[i]), available]  # choose next position from actual allowed positions
            new_poss_pos = np.random.choice(available, p=p)

            if self.adj[position[i], new_poss_pos] == 1:
                position[i + 1] = new_poss_pos
            else:
                position[i + 1] = int(cp.deepcopy(position[i]))

            rel_index, _ = self.relation(position[i], position[i + 1])
            if rel_index < self.n_actions:
                direc[rel_index, i + 1] = 1
                
        return position, direc

    def get_node_positions(self, cells=None, _plot_specs=None, _mask=None):
        xs = []
        ys = []
        self.n_laps = 6
        for i in range(self.n_states):
            pos = i % self.width
            lap = int(i / self.width)
            xs.append(pos)
            ys.append(-lap)

        if cells is not None:
            cell_prepared = np.asarray(cp.deepcopy(cells)).flatten()

            return xs, ys, cell_prepared
        else:
            return xs, ys


def get_new_data_diff_envs(position, pars, envs_class):
    b_s = int(pars.batch_size)
    n_walk = position.shape[-1]  # pars.seq_len
    s_size = pars.s_size

    data = np.zeros((b_s, s_size, n_walk)) 
    for batch in range(b_s):
        data[batch] = sample_data(position[batch, :], envs_class[batch].states_mat, s_size)

    return data


def sample_data(position, states_mat, s_size):
    # makes one-hot encoding of sensory at each time-step
    time_steps = np.shape(position)[0]
    sense_data = np.zeros((s_size, time_steps))
    for i, pos in enumerate(position):
        ind = int(pos)
        sense_data[states_mat[ind], i] = 1
    return sense_data


