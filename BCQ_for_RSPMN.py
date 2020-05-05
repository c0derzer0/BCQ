import argparse
import gym
import numpy as np
import os
import torch

import BCQ
import DDPG
import utils
from collections import defaultdict

# Generates BCQ's replay buffer from numpy dataset format for RSPMNs
def generate_buffer_from_dataset(dataset, vars_in_single_time_step,
                                 sequence_length, state_vars, action_vars,
                                 device, env='test', seed=0, buffer_name='testBuffer'
                                 ):
    # For saving files
    setting = f"{env}_{seed}"
    buffer_name = f"{buffer_name}_{setting}"
    # buffer_name = 'testBuffer'

    # Initialize buffer
    rows = dataset.shape[0]
    replay_buffer = utils.ReplayBuffer(state_vars, action_vars, device,
                                       max_size=rows*sequence_length)
    replay_buffer.size = rows*(sequence_length-1) + 1
    # print(replay_buffer.size)
    replay_buffer.ptr = rows*(sequence_length-1)

    j = 0
    for i in range(0, sequence_length-1):
        if i == 0:
            single_transition = dataset[:, j:j+vars_in_single_time_step+state_vars]
            # print(single_transition)
            # print(single_transition[
            #       :, 0:state_vars
            #       ].reshape(-1, state_vars))
            replay_buffer.state = single_transition[
                                  :, 0:state_vars
                                  ].reshape(-1, state_vars)
            # print(replay_buffer.state)

            replay_buffer.action = single_transition[
                                   :, state_vars:state_vars+action_vars
                                   ].reshape(-1, action_vars)
            replay_buffer.reward = single_transition[
                                   :, state_vars+action_vars
                                   ].reshape(-1,1)
            replay_buffer.next_state = single_transition[
                                       :, state_vars+action_vars+1:
                                       ].reshape(-1, state_vars)
            if i == sequence_length - 2:
                replay_buffer.not_done = np.zeros((rows, 1))

            else:
                replay_buffer.not_done = np.ones((rows, 1))

        else:

            single_transition = dataset[:,
                                j:j + vars_in_single_time_step + state_vars]
            # print(single_transition)
            # print(single_transition[
            #       :, 0:state_vars
            #       ].reshape(-1, state_vars))
            replay_buffer.state = np.concatenate(
                (replay_buffer.state, single_transition[
                                  :, 0:state_vars
                                  ].reshape(-1, state_vars)), axis=0
            )
            # print(replay_buffer.state)

            replay_buffer.action = np.concatenate((replay_buffer.action, single_transition[
                                   :, state_vars:state_vars + action_vars
                                   ].reshape(-1, action_vars)), axis=0)

            replay_buffer.reward = np.concatenate((replay_buffer.reward, single_transition[
                                   :, state_vars + action_vars
                                   ].reshape(-1, 1)), axis=0)

            replay_buffer.next_state = np.concatenate((replay_buffer.next_state, single_transition[
                                       :, state_vars + action_vars + 1:
                                       ].reshape(-1, state_vars)), axis=0)

            if i == sequence_length - 2:
                replay_buffer.not_done = np.concatenate(
                    (replay_buffer.not_done, np.zeros((rows, 1))), axis=0
                )
            else:
                replay_buffer.not_done = np.concatenate(
                    (replay_buffer.not_done, np.ones((rows, 1))), axis=0
                )

        j = j+vars_in_single_time_step

        # # Store data in replay buffer
        # replay_buffer.add(state, action, next_state, reward, done_bool)

    # print(replay_buffer.state)
    if not os.path.exists("./buffers"):
        os.makedirs("./buffers")

    replay_buffer.save(f"./buffers/{buffer_name}")


# Trains BCQ offline
def train_BCQ(state_dim, action_dim, max_action, device, args):

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if not os.path.exists("./models"):
        os.makedirs("./models")

    # For saving files
    setting = f"{args.env}_{args.seed}"
    buffer_name = f"{args.buffer_name}_{setting}"

    # Initialize policy
    policy = BCQ.BCQ(state_dim, action_dim, max_action, device, args.discount,
                     args.tau, args.lmbda, args.phi)

    # Load buffer
    replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device)
    replay_buffer.load(f"./buffers/{buffer_name}")

    evaluations = []
    episode_num = 0
    done = True
    training_iters = 0

    while training_iters < args.max_timesteps:
        pol_vals = policy.train(replay_buffer, iterations=int(args.eval_freq),
                                batch_size=args.batch_size)

        evaluations.append(eval_policy(policy, args.env, args.seed))
        np.save(f"./results/BCQ_{setting}", evaluations)

        training_iters += args.eval_freq
        print(f"Training iterations: {training_iters}")

    return policy


# Trains BCQ offline
def train_DBCQ(dargs, device ):

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if not os.path.exists("./models"):
        os.makedirs("./models")

    # For saving files
    setting = f"{dargs.env}_{dargs.seed}"
    buffer_name = f"{dargs.buffer_name}_{setting}"

    # Initialize policy
    policy = BCQ.BCQ(dargs.parameters, dargs.env_properties, device)

    # Load buffer
    replay_buffer = utils.ReplayBuffer(dargs.parameters["state_dim"],
                                       dargs.parameters["num_actions"], device)
    replay_buffer.load(f"./buffers/{buffer_name}")

    evaluations = []
    episode_num = 0
    done = True
    training_iters = 0

    while training_iters < args.max_timesteps:
        pol_vals = policy.train(replay_buffer)

        evaluations.append(eval_policy(policy, dargs.env, dargs.seed))
        np.save(f"./results/BCQ_{setting}", evaluations)

        training_iters += args.eval_freq
        print(f"Training iterations: {training_iters}")

    return policy


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state))
            action = int(action)
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward


class args:
    def __init__(self, env='Dummy-v0', seed=0, buffer_name='testBuffer',
                 eval_freq=5e3, max_timesteps=1e6, start_timesteps=25e3,
                 rand_action_p=0.3, gaussian_std=0.3, batch_size=100,
                 discount=1, tau=0.005, lmbda=0.75, phi=0.05
                 ):
        self.env = env
        self.seed = seed
        self.buffer_name = buffer_name
        self.eval_freq = eval_freq
        self.max_timesteps = max_timesteps
        self.start_timesteps = start_timesteps
        self.rand_action_p = rand_action_p
        self.gaussian_std = gaussian_std
        self.batch_size = batch_size
        self.discount = discount
        self.tau = tau
        self.lmbda = lmbda
        self.phi = phi

class dargs:

    def __init__(self, num_actions, state_dim,
                 env='Dummy-v0', seed=0, buffer_name='testBuffer',
                 discount=1, eval_freq=5e3, max_timesteps=1e6,
                 target_update_frequency=5e3, update_frequency=5e3,
                 tau=0.005,
                 initial_epsilon=0, end_epsilon=0,
                 epsilon_decay_period=1, evaluation_epsilon=0,
                 optimizer=None, optimizer_parameters=None):
        self.env_properties = defaultdict()
        self.parameters = defaultdict()
        self.env = env
        self.seed = seed
        self.buffer_name = buffer_name
        self.eval_freq = eval_freq
        self.max_timesteps = max_timesteps

        self.env_properties["num_actions"] = num_actions
        self.env_properties["atari"] = False
        self.env_properties["state_dim"] = state_dim

        self.parameters["optimizer"] = optimizer
        self.parameters["optimizer_parameters"] = optimizer_parameters
        self.parameters["discount"] = discount
        self.parameters["target_update_frequency"] = target_update_frequency
        self.parameters["update_frequency"] = update_frequency
        self.parameters["tau"] = tau
        self.parameters["initial_epsilon"] = initial_epsilon
        self.parameters["end_epsilon"] = end_epsilon
        self.parameters["epsilon_decay_period"] = epsilon_decay_period
        self.parameters["evaluation_epsilon"] = evaluation_epsilon











#     print("---------------------------------------")
#
#     if args.generate_buffer_from_dataset:
#         print(f'Setting: Generating buffer from dataset, Env: {args.env}, Seed: {args.seed}')
#         print(f"state_dim {args.state_dim}")
#         print(f"state_dim {args.action_dim}")
#         print(f"state_dim {args.max_action}")
#         print(f"state_dim {args.sequence_length}")
#
#
#     else:
#         print(f"Setting: Training BCQ, Env: {args.env}, Seed: {args.seed}")
#     print("---------------------------------------")
#
#     if not os.path.exists("./results"):
#         os.makedirs("./results")
#
#     if not os.path.exists("./models"):
#         os.makedirs("./models")
#
#     if not os.path.exists("./buffers"):
#         os.makedirs("./buffers")
#
#     torch.manual_seed(args.seed)
#     np.random.seed(args.seed)
#
#     state_dim = args.state_dim
#     action_dim = args.action_dim
#     max_action = args.max_action
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     if args.generate_buffer_from_dataset:
#         generate_buffer_from_dataset(
#             args.dataset, args.vars_in_single_time_step,
#             args.sequence_length,
#             state_dim, action_dim, device, args
#         )
#     else:
#         train_BCQ(state_dim, action_dim, max_action, device, args)
#
#
# if __name__ == "__main__":
#     main()
