#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: James Whittington
"""

import numpy as np
from scipy.special import comb
import itertools
from model_utils import DotDict as Dd


def default_params(width=None, height=None, world_type=None, batch_size=None):
    params = Dd()
    # use graph mode or eager mode
    params.graph_mode = True
    # tf_range=True compiles graph quicker but is slower when running.
    # e.g. for seq_len=75, tf_range=True is 30% slower than tf_range=False, but 80s vs 1000s compilation time
    params.tf_range = False

    params.batch_size = 16 if not batch_size else batch_size
    # seq_len - we truncate BPTT to sequences of this length
    params.seq_len = 10  # 75  # 50
    params.max_states = 30

    params.world_type ='zhao2025' if not world_type else world_type

    # ENVIRONMENT params
    params.n_envs = params.batch_size
    params.s_size = 45 # 45
    params.asyncrounous_envs = True
    params = get_env_params(params, width, height=height)
    params.use_reward = False

    # DATA / SAVE / SUMMARY params params

    # only save date from first X of batch
    params.n_envs_save = 6
    # num gradient updates between summaries
    params.sum_int = 100
    # num gradient updates between detailed accuracy summaries
    params.sum_int_inferences = 400
    # number of gradient steps between saving data
    params.save_interval = int(50000 / params.seq_len)
    # number of gradient steps between saving model
    params.save_model = 5 * params.save_interval

    # MODEL params
    params.infer_g_type = 'g_p'  # 'g'
    params.two_hot = True
    params.s_size_comp = 10 # 10

    # numbers of variables for each frequency
    params.n_grids_all = [20, 20]
    params.grid2phase = 1
    params.n_phases_all = [int(n_grid / params.grid2phase) for n_grid in params.n_grids_all]
    params.tot_phases = sum(params.n_phases_all)
    params.n_freq = len(params.n_phases_all)
    params.g_size = sum(params.n_grids_all)
    params.n_place_all = [p * params.s_size_comp for p in params.n_phases_all]
    params.p_size = sum(params.n_place_all)
    params.s_size_comp_hidden = 20 * params.s_size_comp
    params.prediction_freq = 0

    # These are smoothing parameters, so 'inverse frequencies': higher params.freqs = more smoothing = lower frequency
    params.freqs = sorted([0.01, 0.8])[:params.n_freq]
    params.smooth_only_on_movement = False

    # initialisations
    params.g_init = 0.5
    params.p2g_init = 0.1
    params.g2g_init = 0.2

    # Build matrix for converting one-hot sensory data to two-hot: one_hot * two_hot_mat = two_hot
    params.two_hot_mat = onehot2twohot(np.expand_dims(np.eye(params.s_size), axis=0),
                                       combins_table(params.s_size_comp, 2), params.s_size_comp)

    # TRAINING params
    params.train_iters = 20000
    params.train_on_visited_states_only = True
    params.learning_rate_max = 1e-4
    params.learning_rate_min = 1e-4
    params.logsig_ratio = 6
    params.logsig_offset = 0

    # losses
    params.which_costs = ['lx_p', 'lx_g', 'lx_gt', 'lp', 'lg', 'lg_reg', 'lp_reg']
    if 'p' in params.infer_g_type:
        params.which_costs.append('lp_x')

    # regularisation values
    params.g_reg_pen = 0.0
    params.p_reg_pen = 0.0
    params.weight_reg_val = 0.001

    # Number gradient updates for annealing (in number of gradient updates)
    params.temp_it = 10000
    params.forget_it = 1000
    params.hebb_learn_it = 20000
    params.p2g_use_it = 0
    params.p2g_scale = 100
    params.p2g_sig_val = 50000
    params.g_reg_it = 200000000
    params.p_reg_it = 10000
    params.l_r_decay_steps = 20000
    params.l_r_decay_rate = 0.5

    params.scal_g = 1
    params.scal_p = 1

    # HEBB
    params.hebb_lim = 1
    params.lambd = 0.9999  # 1  # 0.9999
    params.eta = 0.2
    params.hebb_type = [[2], [2]]
    if 'p' not in params.infer_g_type:
        params.hebb_type = [2]

    # Build connectivity matrices for Hebbian weights, which is a list of modules TO, each a list of modules FROM:
    # If connectivity[x][y] is True, that means there is a connection FROM y TO x

    # R_f_F: hierarchical connections within grid modules
    params.R_f_F = connectivity_matrix(conn_hierarchical, params.freqs)
    # R_f_F_inv: all2all connections between and within grid modules
    params.R_f_F_inv = connectivity_matrix(conn_all2all, params.freqs)

    # In the model the place cell mask has the opposite (transpose) meaning from the connectivity matrices
    params.mask_p = get_mask(params.n_place_all, params.n_place_all, transpose_connectivity(params.R_f_F))

    # PLACE ATTRACTOR
    params.kappa = 0.4
    # Overall maximum number of attractor iterations
    params.n_recurs = params.n_freq
    # Set the number of maximum attractor iterations for each module to implement early stopping
    params.max_attractor_its = [params.n_recurs - f for f in range(params.n_freq)]
    params.max_attractor_its_inv = [params.n_recurs for _ in range(params.n_freq)]

    # STATE TRANSITION
    # R_G_F_f: hierarchical connections within grid modules
    params.R_G_F_f = connectivity_matrix(conn_hierarchical, params.freqs) 
    params.mask_g = get_mask(params.n_grids_all, params.n_grids_all, params.R_G_F_f)
    params.d_mixed_size = 20

    # CHANGES SINCE NOW USING AS LITTLE CONTROL FLOW IN GRAPH AS POSSIBLE.
    # The following takes the existing connectivity setup but reorganises it, to optimise for tensorflow performance

    # List of attractor steps, each a list of freqs (in order) that want to update for each attractor step.
    params.attractor_freq_iterations = [[f for f in range(params.n_freq) if r < params.max_attractor_its[f]] for r in
                                        range(params.n_recurs)]
    params.attractor_freq_iterations_inv = [[f for f in range(params.n_freq) if r < params.max_attractor_its_inv[f]] for
                                            r in range(params.n_recurs)]

    # Here each R_f_F is a list (F) of lists (f), with number of frequency 'f' that influence frequency 'F'
    params.R_f_F_ = [list(np.where(x[:params.n_freq])[0]) for x in params.R_f_F]
    params.R_f_F_inv_ = [list(np.where(x[:params.n_freq])[0]) for x in params.R_f_F_inv]

    return params


def get_env_params(par, width, height):
    if par.world_type == 'zhao2025':
        par_env = Dd({'restart_max': 10,
                      'restart_min': 4,
                      'seq_jitter': 2,
                      'save_walk': 30,
                      'sum_inf_walk': 10,
                      'widths': [5 if width is None else width] * par.batch_size,
                      'rels': ['run', 'stay still']
                      })
    else:
        print(par.world_type)
        raise ValueError('incorrect world specified')

    par_env.n_actions = len(par_env.rels) if 'stay still' not in par_env.rels else len(par_env.rels) - 1
    par.n_actions = par_env.n_actions
    par.env = par_env
    return par

def get_scaling_parameters(index, par):
    min_scale = 1.0
    # these scale with number of gradient updates
    temp = np.minimum((index + 1) / par.temp_it, min_scale)
    forget = np.minimum((index + 1) / par.forget_it, min_scale)
    hebb_learn = np.minimum((index + 1) / par.hebb_learn_it, min_scale)
    p2g_use = sigmoid((index - par.p2g_use_it) / par.p2g_scale)
    l_r = (par.learning_rate_max - par.learning_rate_min) * (par.l_r_decay_rate ** (
            index / par.l_r_decay_steps)) + par.learning_rate_min
    l_r = np.maximum(l_r, par.learning_rate_min)
    g_cell_reg = 1 - np.minimum((index + 1) / par.g_reg_it, min_scale)
    p_cell_reg = 1 - np.minimum((index + 1) / par.p_reg_it, min_scale)

    scalings = Dd({'temp': temp,
                   'forget': forget,
                   'h_l': hebb_learn,
                   'p2g_use': p2g_use,
                   'l_r': l_r,
                   'g_cell_reg': g_cell_reg,
                   'p_cell_reg': p_cell_reg,
                   'iteration': index,
                   })

    return scalings


def connectivity_matrix(g2g, freqs):
    """
    Build connectivity matrices between modules. C is a list of modules TO, each a list of modules FROM:
    If C[x][y] is True, that means there is a connection FROM y TO x
    g2g are functions that return whether a connection exists, given the 'frequency'
    (actually, exponential smoothing - so more like inverse frequency) of both modules
    """
    connec = [[None for _ in range(len(freqs))] for _ in range(len(freqs))]
    for f_from in range(len(freqs)):
        for f_to in range(len(freqs)):
            connec[f_to][f_from] = g2g(freqs[f_from], freqs[f_to])
    return connec


def transpose_connectivity(connec):
    """
    C is a list of modules TO, each a list of modules FROM: if C[x][y] is True, that means there is a connection 
    FROM y TO x. This function calculates the transpose, collecting the ith entry of each input inner list 
    in the ith output inner list. Thus if C_T[x][y] is True, there is a connection FROM x TO y.
    """
    connec_t = [list(entry_i) for entry_i in zip(*connec)]
    return connec_t


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_mask(n_cells_in, n_cells_out, r):
    """
    Generate a mask matrix M_ij that for each cell i holds if it recieves input from cell (i.e. connection from j to i)
    Input a list of cells per module and a connectivity matrix r_ij, which is list of lists that indicates the
    connectivity from module j to i: if r[i][j] is True, then module i recieves input from module j
    """

    n_freq = len(n_cells_in)
    n_all_in = sum(n_cells_in)
    n_all_out = sum(n_cells_out)
    c_p_in = np.insert(np.cumsum(n_cells_in), 0, 0).astype(int)
    c_p_out = np.insert(np.cumsum(n_cells_out), 0, 0).astype(int)

    mask = np.zeros((n_all_in, n_all_out), dtype=np.float32)

    for f_to in range(n_freq):
        for f_from in range(n_freq):
            mask[c_p_in[f_to]:c_p_in[f_to + 1], c_p_out[f_from]:c_p_out[f_from + 1]] = r[f_to][f_from]

    return mask


def combins(n, k, m):
    s = []
    for i in range(1, n + 1):
        c = comb(n - i, k)
        if m >= c:
            s.append(1)
            m -= c
            k = k - 1
        else:
            s.append(0)
    return tuple(s)


def combins_table(n, k, map_max=None):
    table = []
    rev_table = {}
    table_top = comb(n, k)
    for m in range(int(table_top)):
        # forward mapping
        c = combins(n, k, m)
        if map_max is None or m < map_max:
            table.append(c)
            rev_table[c] = m
        else:
            rev_table[c] = m % map_max
    return table


def onehot2twohot(onehot, table, compress_size):
    seq_len = np.shape(onehot)[2]
    batch_size = np.shape(onehot)[0]
    twohot = np.zeros((batch_size, compress_size, seq_len))
    for i in range(np.shape(onehot)[2]):
        vals = np.argmax(onehot[:, :, i], 1)
        for b in range(np.shape(onehot)[0]):
            twohot[b, :, i] = table[vals[int(b)]]

    return twohot


# Specify types of connections between modules, from initial values of exponential smoothing a ('inverse frequency')
def conn_hierarchical(a_from, a_to):
    return int(a_from >= a_to)  # Allow connections only from low to high frequency


def conn_separate(a_from, a_to):
    return int(a_from == a_to)  # Allow connections only within frequency


def conn_all2all(*_):
    return int(True)  # Allow all connections, independent of frequencies of modules


def conn_none2none(*_):
    return int(False)  # Allow no connections at all, independent of frequencies


def old2new(world_type):
    old2new_name_convert = Dd({
                               'zhao2025': 'zhao2025'
                               })
    try:
        return old2new_name_convert[world_type]
    except KeyError:
        return world_type
