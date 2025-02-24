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

import numpy as np

from src.reinforce import REINFORCE


class TestLinearFeature:
    def __init__(self, feature_matrix):
        self.feature_matrix = feature_matrix

    @property
    def n_states(self):
        return self.feature_matrix.shape[0]
    
    @property
    def n_actions(self):
        return self.feature_matrix.shape[1]

    @property
    def dim(self):
        return self.feature_matrix.shape[-1]
    
    def __call__(self, state, action):
        return self.feature_matrix[state, action]


def get_reinforce_results(shared_data):
    results = dict()
    for case_i in shared_data:
        feat = TestLinearFeature(shared_data[case_i]["features"])
        params = shared_data[case_i]["params"]
        trajs = shared_data[case_i]["trajs"]

        results[case_i] = {
            "get_action": [],
            "compute_update": [],
        }

        # Test get_action
        agent = REINFORCE(
            feat.n_actions,
            feat,
            np.random.RandomState(42),
            np.random,
            None,
        )
        agent.parameters = np.copy(params)

        for traj_curr_states in trajs["curr_states"]:
            for curr_state in traj_curr_states:
                results[case_i]["get_action"].append(
                    agent.get_action(curr_state)
                )

        # Test compute_update
        for alpha in [0.1, 0.5, 0.9]:
            agent = REINFORCE(
                feat.n_actions,
                feat,
                np.random.RandomState(42),
                np.random,
                alpha,
            )
            agent.parameters = np.copy(params)

            for traj in zip(
                trajs["curr_states"],
                trajs["actions"],
                trajs["rewards"],
                trajs["dones"],
                trajs["next_states"],
            ):
                batch = list(zip(*traj))
                for step_i in range(len(traj)):
                    results[case_i]["compute_update"].append(
                        agent.compute_update(batch, step_i)
                    )
    return results


if __name__ == "__main__":
    import _pickle as pickle

    test_data = pickle.load(
        open("{}/test_data.pkl".format(currentdir), "rb")
    )

    shared_data = test_data["shared_data"]
    results = get_reinforce_results(shared_data)
    
    expected_results = test_data["reinforce"]

    assert len(results) == len(expected_results)

    all_matches = dict()
    for test_method in [
        "get_action",
        "compute_update",
    ]:
        all_matches.setdefault(test_method, True)
        for case_i in results:
            result = results[case_i][test_method]
            expected_result = expected_results[case_i][test_method]

            try:
                is_match = np.allclose(
                    result,
                    expected_result
                )
                if not is_match:
                    print(
                        "Test case {}: {} failed---values are not close enough".format(case_i, test_method)
                    )
            except:
                is_match = False
                print(
                    "Test case {}: {} failed---unable to compare results, likely due to shape mismatch".format(case_i, test_method)
                )

            all_matches[test_method] = all_matches[test_method] and is_match

    print("Correct implementation: {}".format(all_matches))
        
