import argparse
import copy
import importlib
import json
import os
from collections import defaultdict

import numpy as np
import torch

import discrete_BCQ
import DQN
import utils


def interact_with_environment(env, replay_buffer, is_atari, state_dim,
                              num_actions, args, parameters, device):
    # For saving files
    setting = f"{args.env}_{args.seed}"
    buffer_name = f"{args.buffer_name}_{setting}"

    # Initialize and load policy
    policy = DQN.DQN(
        is_atari,
        num_actions,
        state_dim,
        device,
        parameters["discount"],
        parameters["optimizer"],
        parameters["optimizer_parameters"],
        parameters["polyak_target_update"],
        parameters["target_update_freq"],
        parameters["tau"],
        parameters["initial_eps"],
        parameters["end_eps"],
        parameters["eps_decay_period"],
        parameters["eval_eps"],
    )

    if args.generate_buffer: policy.load(f"./models/behavioral_{setting}")

    evaluations = []

    state, done = env.reset(), False
    episode_start = True
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    low_noise_ep = np.random.uniform(0, 1) < args.low_noise_p
    eval_policy(policy, args.env, args.seed, eval_episodes=1000)
    # Interact with the environment for max_timesteps
    for t in range(int(args.max_timesteps)):

        episode_timesteps += 1

        # If generating the buffer, episode is low noise with p=low_noise_p.
        # If policy is low noise, we take random actions with p=eval_eps.
        # If the policy is high noise, we take random actions with p=rand_action_p.
        if args.generate_buffer:
            if not low_noise_ep and np.random.uniform(0,
                                                      1) < args.rand_action_p - \
                    parameters["eval_eps"]:
                action = env.action_space.sample()
            else:
                # print("state in interact_with_environment:\t"+str(state))
                action = policy.select_action(np.array(state), eval=True)

        if args.train_behavioral:
            if t < parameters["start_timesteps"]:
                action = env.action_space.sample()
            else:
                # print("state in interact_with_environment:\t"+str(state))
                action = policy.select_action(np.array(state))

        # Perform action and log results
        next_state, reward, done, info = env.step(action)
        episode_reward += reward

        # Only consider "done" if episode terminates due to failure condition
        done_float = float(
            done) if episode_timesteps < env._max_episode_steps else 0

        # For atari, info[0] = clipped reward, info[1] = done_float
        if is_atari:
            reward = info[0]
            done_float = info[1]

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_float, done,
                          episode_start)
        state = copy.copy(next_state)
        episode_start = False

        # Train agent after collecting sufficient data
        if args.train_behavioral and t >= parameters["start_timesteps"] and (
                t + 1) % parameters["train_freq"] == 0:
            policy.train(replay_buffer)

        if done:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(
                f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            # Reset environment
            state, done = env.reset(), False
            episode_start = True
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
            low_noise_ep = np.random.uniform(0, 1) < args.low_noise_p

        # Evaluate episode
        if args.train_behavioral and (t + 1) % parameters["eval_freq"] == 0:
            evaluations.append(eval_policy(policy, args.env, args.seed))
            np.save(f"./results/behavioral_{setting}", evaluations)
            policy.save(f"./models/behavioral_{setting}")

    # Save final policy
    if args.train_behavioral:
        policy.save(f"./models/behavioral_{setting}")

    # Save final buffer and performance
    else:
        evaluations.append(eval_policy(policy, args.env, args.seed))
        np.save(f"./results/buffer_performance_{setting}", evaluations)
        replay_buffer.save(f"./buffers/{buffer_name}")


# Generates BCQ's replay buffer from numpy dataset format for RSPMNs
def generate_buffer_from_dataset(dataset, vars_in_single_time_step,
                                 sequence_length, state_vars, action_vars,
                                 device, env='test', seed=0,
                                 buffer_name='testBuffer'
                                 ):
    # For saving files
    setting = f"{env}_{seed}"
    buffer_name = f"{buffer_name}_{setting}"
    # buffer_name = 'testBuffer'

    # Initialize buffer
    rows = dataset.shape[0]
    replay_buffer = utils.StandardBuffer(state_vars, action_vars,
                                       rows * sequence_length,
                                       rows * sequence_length,
                                       device
                                       )
    replay_buffer.crt_size = rows * (sequence_length - 1) + 1
    # print(replay_buffer.size)
    replay_buffer.ptr = rows * (sequence_length - 1)

    j = 0
    for i in range(0, sequence_length - 1):
        if i == 0:
            single_transition = dataset[:,
                                j:j + vars_in_single_time_step + state_vars]
            # print(single_transition)
            # print(single_transition[
            #       :, 0:state_vars
            #       ].reshape(-1, state_vars))
            replay_buffer.state = single_transition[
                                  :, 0:state_vars
                                  ].reshape(-1, state_vars)
            # print(replay_buffer.state)

            replay_buffer.action = single_transition[
                                   :, state_vars:state_vars + action_vars
                                   ].reshape(-1, action_vars)
            replay_buffer.reward = single_transition[
                                   :, state_vars + action_vars
                                   ].reshape(-1, 1)
            replay_buffer.next_state = single_transition[
                                       :, state_vars + action_vars + 1:
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

            replay_buffer.action = np.concatenate(
                (replay_buffer.action, single_transition[
                                       :, state_vars:state_vars + action_vars
                                       ].reshape(-1, action_vars)), axis=0)

            replay_buffer.reward = np.concatenate(
                (replay_buffer.reward, single_transition[
                                       :, state_vars + action_vars
                                       ].reshape(-1, 1)), axis=0)

            replay_buffer.next_state = np.concatenate(
                (replay_buffer.next_state, single_transition[
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

        j = j + vars_in_single_time_step

        # # Store data in replay buffer
        # replay_buffer.add(state, action, next_state, reward, done_bool)

    # print(replay_buffer.state)
    if not os.path.exists("./buffers"):
        os.makedirs("./buffers")

    replay_buffer.save(f"./buffers/{buffer_name}")


# Trains BCQ offline
def train_BCQ(env, replay_buffer, is_atari, state_dim, num_actions, args,
              parameters=None, device=None):
    # For saving files
    setting = f"{args.env}_{args.seed}"
    buffer_name = f"{args.buffer_name}_{setting}"

    # # Initialize and load policy
    # policy = discrete_BCQ.discrete_BCQ(
    #     is_atari,
    #     num_actions,
    #     state_dim,
    #     device,
    #     args.BCQ_threshold,
    #     parameters["discount"],
    #     parameters["optimizer"],
    #     parameters["optimizer_parameters"],
    #     parameters["polyak_target_update"],
    #     parameters["target_update_freq"],
    #     parameters["tau"],
    #     parameters["initial_eps"],
    #     parameters["end_eps"],
    #     parameters["eps_decay_period"],
    #     parameters["eval_eps"]
    # )

    policy = discrete_BCQ.discrete_BCQ(
        is_atari,
        num_actions,
        state_dim,
        device,
        args.BCQ_threshold,
        args.parameters["discount"],
        args.parameters["optimizer"],
        args.parameters["optimizer_parameters"],
        args.parameters["polyak_target_update"],
        args.parameters["target_update_freq"],
        args.parameters["tau"],
        args.parameters["initial_eps"],
        args.parameters["end_eps"],
        args.parameters["eps_decay_period"],
        args.parameters["eval_eps"]
    )

    # Load replay buffer
    replay_buffer.load(f"./buffers/{buffer_name}")

    evaluations = []
    episode_num = 0
    done = True
    training_iters = 0

    while training_iters < args.max_timesteps:

        for _ in range(int(parameters["eval_freq"])):
            policy.train(replay_buffer)

        if args.do_eval_policy:
            evaluations.append(eval_policy(policy, args.env, args.seed))
            np.save(f"./results/BCQ_{setting}", evaluations)

            training_iters += int(parameters["eval_freq"])
            print(f"Training iterations: {training_iters}")

    return policy


def train_DBCQ(args,
               device):
    # For saving files
    setting = f"{args.env}_{args.seed}"
    buffer_name = f"{args.buffer_name}_{setting}"

    policy = discrete_BCQ.discrete_BCQ(
        args.env_properties["atari"],
        args.env_properties["num_actions"],
        args.env_properties["state_dim"],
        device,
        args.BCQ_threshold,
        args.parameters["discount"],
        args.parameters["optimizer"],
        args.parameters["optimizer_parameters"],
        args.parameters["polyak_target_update"],
        args.parameters["target_update_freq"],
        args.parameters["tau"],
        args.parameters["initial_eps"],
        args.parameters["end_eps"],
        args.parameters["eps_decay_period"],
        args.parameters["eval_eps"]
    )

    # Load replay buffer
    replay_buffer = utils.StandardBuffer(args.env_properties["state_dim"],
                                         args.env_properties["action_dim"],
                                         args.parameters['batch_size'],
                                         args.parameters['buffer_size'],
                                         device
                                         )
    replay_buffer.load(f"./buffers/{buffer_name}")

    evaluations = []
    episode_num = 0
    done = True
    training_iters = 0

    while training_iters < args.max_timesteps:

        for _ in range(int(args.parameters["eval_freq"])):
            policy.train(replay_buffer)

        if args.do_eval_policy:
            evaluations.append(eval_policy(policy, args.env, args.seed))
            np.save(f"./results/BCQ_{setting}", evaluations)

            training_iters += int(args.parameters["eval_freq"])
            print(f"Training iterations: {training_iters}")

    return policy


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):
    eval_env, _, _, _ = utils.make_env(env_name, atari_preprocessing)
    eval_env.seed(seed + 100)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state), eval=True)
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward


class dargs:

    def __init__(self, num_actions, state_dim, action_dim,
                 env='Dummy-v0', seed=0, buffer_name='testBuffer',
                 # Exploration
                 start_timesteps=1e3, initial_epsilon=0.1,
                 end_epsilon=0, epsilon_decay_period=1,
                 # Evaluation
                 eval_freq=5e3, evaluation_epsilon=0,
                 # Learning
                 discount=1, buffer_size=1e6,
                 batch_size=64, optimizer="Adam",
                 optimizer_parameters=None,
                 train_freq=1, polyak_target_update=True,
                 target_update_frequency=1, tau=0.005,
                 # misc
                 max_timesteps=1e6, BCQ_threshold=0.3,
                 do_eval_policy = False,
                 low_noise_p=0.2, rand_action_p=0.2,
                 ):
        self.env_properties = defaultdict()
        self.parameters = defaultdict()
        self.env = env
        self.seed = seed
        self.buffer_name = buffer_name

        self.max_timesteps = max_timesteps
        self.BCQ_threshold = BCQ_threshold
        self.do_eval_policy = do_eval_policy

        # regular_parameters = {
        #
        #     "start_timesteps": 1e3,
        #     "initial_eps": 0.1,
        #     "end_eps": 0.1,
        #     "eps_decay_period": 1,
        #
        #     "eval_freq": 5e3,
        #     "eval_eps": 0,
        #     # Learning
        #     "discount": 0.99,
        #     "buffer_size": 1e6,
        #     "batch_size": 64,
        #     "optimizer": "Adam",
        #     "optimizer_parameters": {
        #         "lr": 3e-4
        #     },
        #     "train_freq": 1,
        #     "polyak_target_update": True,
        #     "target_update_freq": 1,
        #     "tau": 0.005
        # }

        self.env_properties["num_actions"] = num_actions
        self.env_properties["atari"] = False
        self.env_properties["state_dim"] = state_dim
        self.env_properties["action_dim"] = action_dim

        self.parameters["optimizer"] = optimizer
        self.parameters["optimizer_parameters"] = optimizer_parameters
        self.parameters["discount"] = discount
        self.parameters["target_update_freq"] = target_update_frequency
        self.parameters["start_timesteps"] = start_timesteps
        self.parameters["tau"] = tau
        self.parameters["initial_eps"] = initial_epsilon
        self.parameters["end_eps"] = end_epsilon
        self.parameters["eps_decay_period"] = epsilon_decay_period
        self.parameters["eval_eps"] = evaluation_epsilon
        self.parameters["polyak_target_update"] = polyak_target_update
        self.parameters['batch_size'] = batch_size
        self.parameters['buffer_size'] = buffer_size
        self.parameters['eval_freq'] = eval_freq
        self.parameters['train_freq'] = train_freq





if __name__ == "__main__":

    # Atari Specific
    atari_preprocessing = {
        "frame_skip": 4,
        "frame_size": 84,
        "state_history": 4,
        "done_on_life_loss": False,
        "reward_clipping": True,
        "max_episode_timesteps": 27e3
    }

    atari_parameters = {
        # Exploration
        "start_timesteps": 2e4,
        "initial_eps": 1,
        "end_eps": 1e-2,
        "eps_decay_period": 25e4,
        # Evaluation
        "eval_freq": 5e4,
        "eval_eps": 1e-3,
        # Learning
        "discount": 0.99,
        "buffer_size": 1e6,
        "batch_size": 32,
        "optimizer": "Adam",
        "optimizer_parameters": {
            "lr": 0.0000625,
            "eps": 0.00015
        },
        "train_freq": 4,
        "polyak_target_update": False,
        "target_update_freq": 8e3,
        "tau": 1
    }

    regular_parameters = {
        # Exploration
        "start_timesteps": 1e3,
        "initial_eps": 0.1,
        "end_eps": 0.1,
        "eps_decay_period": 1,
        # Evaluation
        "eval_freq": 5e3,
        "eval_eps": 0,
        # Learning
        "discount": 0.99,
        "buffer_size": 1e6,
        "batch_size": 64,
        "optimizer": "Adam",
        "optimizer_parameters": {
            "lr": 3e-4
        },
        "train_freq": 1,
        "polyak_target_update": True,
        "target_update_freq": 1,
        "tau": 0.005
    }

    # Load parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",
                        default="PongNoFrameskip-v0")  # OpenAI gym environment name
    parser.add_argument("--seed", default=0,
                        type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--buffer_name",
                        default="Default")  # Prepends name to filename
    parser.add_argument("--max_timesteps", default=1e6,
                        type=int)  # Max time steps to run environment or train for
    parser.add_argument("--BCQ_threshold", default=0.3,
                        type=float)  # Threshold hyper-parameter for BCQ
    parser.add_argument("--low_noise_p", default=0.2,
                        type=float)  # Probability of a low noise episode when generating buffer
    parser.add_argument("--rand_action_p", default=0.2,
                        type=float)  # Probability of taking a random action when generating buffer, during non-low noise episode
    parser.add_argument("--train_behavioral",
                        action="store_true")  # If true, train behavioral policy
    parser.add_argument("--generate_buffer",
                        action="store_true")  # If true, generate buffer
    parser.add_argument("--state_dim",
                        default=1)
    parser.add_argument("--action_dim",
                        default=1)

    args = parser.parse_args()

    print("---------------------------------------")
    if args.train_behavioral:
        print(
            f"Setting: Training behavioral, Env: {args.env}, Seed: {args.seed}")
    elif args.generate_buffer:
        print(f"Setting: Generating buffer, Env: {args.env}, Seed: {args.seed}")
    else:
        print(f"Setting: Training BCQ, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if args.train_behavioral and args.generate_buffer:
        print("Train_behavioral and generate_buffer cannot both be true.")
        exit()

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if not os.path.exists("./models"):
        os.makedirs("./models")

    if not os.path.exists("./buffers"):
        os.makedirs("./buffers")

    # Make env and determine properties
    env, is_atari, state_dim, num_actions = utils.make_env(args.env,
                                                           atari_preprocessing)
    parameters = atari_parameters if is_atari else regular_parameters

    action_dim = parameters['action_dim']
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize buffer
    replay_buffer = utils.ReplayBuffer(state_dim, action_dim, is_atari,
                                       atari_preprocessing,
                                       parameters["batch_size"],
                                       parameters["buffer_size"], device)

    if args.train_behavioral or args.generate_buffer:
        interact_with_environment(env, replay_buffer, is_atari, state_dim,
                                  num_actions, args, parameters, device)
    else:
        train_BCQ(env, replay_buffer, is_atari, state_dim, num_actions, args,
                  parameters, device)
