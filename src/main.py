"""
CSCD84 - Artificial Intelligence, Winter 2025, Assignment 2
B. Chan
"""

import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from datetime import datetime

import _pickle as pickle
import argparse
import numpy as np

from src.env import FrozenLakeEnv
from src.features import LinearFeature, TabularFeature, ReferenceLinearFeature
from src.q_learning import QLearning
from src.reinforce import REINFORCE


class RandomAgent:
    def __init__(self, action_space, policy_rng):
        self.action_space = action_space
        self.policy_rng = policy_rng
        self.parameters = None

    def get_action(self, *args, **kwargs):
        self.action_space.seed(self.policy_rng.randint(2 ** 10))
        return self.action_space.sample()
    
    def learn(self, *args, **kwargs):
        pass


def main(args):
    map_size = args.map_size
    horizon = args.horizon
    env_seed = args.env_seed
    sample_seed = args.sample_seed
    algorithm = args.algorithm
    feature_type = args.feature_type
    num_trajs = args.num_trajs
    alpha = args.alpha
    eps = args.eps
    disable_render = args.disable_render
    train = not args.evaluate
    load_params = args.load_params

    trajectory_rng = np.random.RandomState(sample_seed)
    policy_rng = np.random.RandomState(sample_seed)
    learner_rng = np.random.RandomState(sample_seed)

    env = FrozenLakeEnv(
        horizon=horizon,
        map_size=map_size,
        seed=env_seed,
        render_mode=None if disable_render else "human"
    )

    num_actions = env.action_space.n
    if feature_type == "tabular":
        feature = TabularFeature()
    elif feature_type == "linear":
        feature = LinearFeature()
    elif feature_type == "linear_ref":
        feature = ReferenceLinearFeature()
    else:
        raise ValueError("feature_type {} not supported".format(feature_type))

    if algorithm == "q_learning":
        agent = QLearning(
            num_actions,
            feature,
            policy_rng,
            learner_rng,
            alpha,
            eps,
        )
    elif algorithm == "reinforce":
        agent = REINFORCE(
            num_actions,
            feature,
            policy_rng,
            learner_rng,
            alpha,
        )
    elif algorithm == "random":
        agent = RandomAgent(
            env.action_space,
            policy_rng,
        )
    else:
        raise ValueError(
            "No such algorithm {} implemented".format(algorithm)
        )

    if load_params is not None and os.path.isfile(load_params):
        agent.parameters = pickle.load(open(load_params, "rb"))
        print("Loaded model parameters from {}".format(load_params))

    ep_returns = []
    reach_goals = []
    for traj_i in range(num_trajs):
        done = False
        ep_returns.append(0)
        reach_goals.append(False)

        curr_state, _ = env.reset(seed=trajectory_rng.randint(2 ** 10))
        while not done:
            action = agent.get_action(curr_state)
            next_state, reward, done, _, _ = env.step(action)

            if train:
                agent.learn(curr_state, action, reward, done, next_state)
            curr_state = next_state

            ep_returns[-1] += reward
            reach_goals[-1] = reach_goals[-1] or reward > 0

        if (traj_i + 1) % 100 == 0:
            print("Average return for last 100 trajectories: {} +/- {}".format(
                np.mean(ep_returns[-100:]),
                np.std(ep_returns[-100:]),
            ))
            print("Num trajectories passing through treasure: {}".format(
                np.sum(reach_goals[-100:]),
            ))

    if train:
        time_tag = datetime.strftime(datetime.now(), "%m-%d-%y_%H_%M_%S")
        save_path = "params-{}-{}-{}.pkl".format(
            algorithm,
            feature_type,
            time_tag
        )
        pickle.dump(
            agent.parameters,
            open(save_path, "wb")
        )
        print("Saved model parameters to {}".format(save_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--map_size",
        type=int,
        default=8,
        help="The size of the map"
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=50,
        help="The length of the trajectory (or horizon)"
    )

    parser.add_argument(
        "--env_seed",
        type=int,
        default=0,
        help="The random seed to use for the environment"
    )

    parser.add_argument(
        "--sample_seed",
        type=int,
        default=0,
        help="The random seed to use for the sampling, including parameter initialization, action sampling, and trajectory seeding"
    )

    parser.add_argument(
        "--algorithm",
        choices=[
            "random",
            "q_learning",
            "reinforce",
        ],
        default="random",
        help="The algorithm to use"
    )

    parser.add_argument(
        "--feature_type",
        choices=[
            "tabular",
            "linear",
            "linear_ref"
        ],
        help="Feature to use"
    )

    parser.add_argument(
        "--num_trajs",
        type=int,
        default=10000,
        help="The number of trajectories"
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=1e-2,
        help="The learning rate, alpha, for training"
    )

    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Whether or not to evaluate agent"
    )

    parser.add_argument(
        "--disable_render",
        action="store_true",
        help="Whether or not to disable rendering---this will speed up the code"
    )

    parser.add_argument(
        "--load_params",
        type=str,
        default=None,
        help="Whether or not to load a trained agent"
    )

    # Q-learning specific

    parser.add_argument(
        "--eps",
        type=float,
        default=0.1,
        help="Epsilon in epsilon-greedy exploration"
    )


    args = parser.parse_args()
    main(args)
