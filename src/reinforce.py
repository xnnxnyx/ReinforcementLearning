"""
CSCD84 - Artificial Intelligence, Winter 2025, Assignment 2
B. Chan
"""

import numpy as np


def softmax(x):
    """
    Compute softmax values for each sets of scores in x.
    Note that softmax is shift invariant,
    hence our implementation shifts by the max logits to mitigate numerical instability.
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


class REINFORCE:
    def __init__(self, num_actions, features, policy_rng, learner_rng, alpha):
        self.num_actions = num_actions
        self.features = features
        self.policy_rng = policy_rng
        self.learner_rng = learner_rng
        self.alpha = alpha

        # Randomly initialize parameters following standard normal
        self.parameters = learner_rng.randn(self.features.dim)

        # This keeps track of the current trajectory
        self.curr_traj = []

    def get_action(self, curr_state, *args, **kwargs):
        """
        Samples action using linear softmax policy.
        """

        # NOTE: You should be modifying prob_dist in the TODO block.
        # prob_dist = np.ones(self.num_actions) / self.num_actions

        # ========================================================
        # TODO: Compute the probability distribution over actions given the current state
        logits = np.array([self.parameters @ self.features(curr_state, a) for a in range(self.num_actions)])
        prob_dist = softmax(logits)
        # ========================================================

        action = self.policy_rng.choice(self.num_actions, p=prob_dist)

        return int(action)

    def compute_update(self, traj, timestep):
        """
        Computes the update for timestep t

        Recall that the update rule at timestep t is:
            G_t = sum_{k=t}^{T - 1} R_k
            update = G_t * grad_w log(pi_w(A_t | S_t))

        traj is a list:
        [
            (S_0, A_0, R_0, DONE_0, S_1),
            (S_1, A_1, R_1, DONE_1, S_2),
            ...
            (S_{T-1}, A_{T-1}, R_{T-1}, DONE_{T-1}, S_T),
        ]
        """

        traj_len = len(traj)
        update = None

        # ========================================================
        # TODO: Implement the update rule
        G_t = sum(traj[t][2] for t in range(timestep, traj_len))
        curr_state, action, _, _, _ = traj[timestep]

        feature = self.features(curr_state, action)
        logits = np.array([self.parameters @ self.features(curr_state, a) for a in range(self.num_actions)])
        prob_dist = softmax(logits)
        grad_log = feature - sum(prob_dist[a] * self.features(curr_state, a) for a in range(self.num_actions))
        update = G_t * grad_log
        # ========================================================

        return update

    def learn(self, curr_state, action, reward, done, next_state, *args, **kwargs):
        """
        Updates the parameters of the Q-function
        """

        self.curr_traj.append(
            (curr_state, action, reward, done, next_state)
        )

        if done:
            for timestep in np.arange(len(self.curr_traj))[::-1]:
                update = self.compute_update(self.curr_traj, timestep)
                self.parameters += self.alpha * update

            # Empty out memory because we expect a new trajectory
            self.curr_traj = []
