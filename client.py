import numpy as np
import time
import argparse
import pickle
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import entropy

import pyemma
import math
from os.path import join
from pythonosc import osc_message_builder
from pythonosc import udp_client

# root_dir = '/Users/robert_arbon/Google Drive/Research/Sonification/'
# data_dir = join(root_dir, 'Data/Chodera_data/Processed')
#
# # Trajectories
# entropy = np.load(join(data_dir, 'shannon_entropy_traj_lag1.0ps.npy'))
# fast_modes = np.load(join(data_dir, 'fast_modes_lag1.0ps.npy'))
# hmm_traj = np.load(join(data_dir, 'probabilistic_traj_lag1.0ps.npy'))
# free_energy_traj = np.load(join(data_dir, 'free_energy_traj_lag1.0ps.npy'))
#
# # Static properties
# properties = np.load(join(data_dir, 'static_properties_max_scale.pickle'))
#
# # Take derivatives
# dfast_modes = fast_modes[:-1]-fast_modes[1:]
# hmm_traj = hmm_traj[1:]
# entropy = entropy[1:]
# free_energy_traj = free_energy_traj[1:]
#
# Nsteps = hmm_traj.shape[0]


def get_traj_info(model):
    """
    returns the number and length of each trajectory
    :param model:
    :return:
    """
    traj_obs = hmm.discrete_trajectories_obs
    n_trajs_obs = len(traj_obs)
    traj_lengths = [x.shape[0] for x in traj_obs]
    print('Number of trajectories: {}'.format(n_trajs_obs))
    print('Lengths of trajectories: {}'.format(traj_lengths))
    return n_trajs_obs, traj_lengths


def load_hmm(path):
    """
    loads the HMM from the given path
    :param path:
    :return:
    """
    print('loading HMM object')
    hmm = pickle.load(open(path, 'rb'))
    print(hmm)
    return hmm


def get_free_energy_traj(model, idx=0):
    # Get stationary distribution and convert to Free Energy
    stat_dist = model.stationary_distribution_obs
    free_energy = -np.log(stat_dist)
    # Invert the scale
    free_energy = np.abs(np.max(free_energy)-free_energy)
    # Calculate free energy of each state
    traj = hmm.discrete_trajectories_obs[idx]
    free_energy_traj = free_energy[traj]
    # Scale to between 0 - 1
    scaler = MinMaxScaler()
    free_energy_traj_scaled = scaler.fit_transform(free_energy_traj[:, np.newaxis])
    return free_energy_traj_scaled


def get_entropy_traj(model, idx=0):
    p_traj = model.hidden_state_probabilities[idx]
    entropy_traj = entropy(p_traj.T)
    return entropy_traj


def get_param_trajs(model, idx=0):
    return {'free_energy': get_free_energy_traj(model),
            'entropy': get_entropy_traj(model),
            'state': model.hidden_state_probabilities[idx]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="127.0.0.1",
                        help="the IP address to broadcast to")
    parser.add_argument("--port", type=int, default=5005,
                        help="the port to broadcast to")
    parser.add_argument("--hmm",
                        help="path to a pickled PyEMMA HMM object")
    parser.add_argument("--delay", type=float, default=0.05,
                        help="delay after every frame of the trajectory")
    parser.add_argument("--traj-index", type=int, default=0,
                        help="the index of the trajectory you want to broadcast")
    parser.add_argument("--loop", type=bool, default=True,
                        help="whether to loop continuously")
    args = parser.parse_args()

    # Get data
    hmm = load_hmm(args.hmm)

    n_trajs, traj_lens = get_traj_info(hmm)

    traj_idx = args.traj_index

    param_trajs = get_param_trajs(hmm, idx=traj_idx)

    # Setup client
    client = udp_client.SimpleUDPClient(args.ip, args.port)

    # for k, v in properties.items():
    #     for i, x in enumerate(v):
    #         client.send_message("/properties/{0}/state{1}".format(k,i), float(x))
    #

    # Broadcast parameters
    n_steps = traj_lens[traj_idx]
    n_states = hmm.n_states
    delay = args.delay

    # Broadcast loop
    # todo make this respect the loop argument
    for i in range(n_steps):

        for j in range(n_states):
            client.send_message("/state{}".format(j+1),
                                float(param_trajs['state'][i][j]))

        client.send_message("/free_energy",
                            float(param_trajs['free_energy'][i]))

        client.send_message("/entropy", float(param_trajs['free_energy'][i]))

        time.sleep(delay)
