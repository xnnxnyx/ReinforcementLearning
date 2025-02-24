"""
CSCD84 - Artificial Intelligence, Winter 2025, Assignment 2
B. Chan
"""

import numpy as np


class TabularFeature:
    """
    Tabular feature
    """

    @property
    def dim(self):
        return 64 * 5
    
    def __call__(self, state, action):
        q_table = np.zeros((64, 5))

        char_loc = np.where(state == b"C")
        char_loc_flattened = (char_loc[0] * 8 + char_loc[1]).item()
        
        q_table[char_loc_flattened, action] = 1

        return q_table.flatten()


class ReferenceLinearFeature:
    """
    A reference linear feature---it is by no means the best features.
    """

    @property
    def dim(self):
        return 64 * 2 + 1
    
    def __call__(self, state, action):
        char_loc = np.where(state == b"C")
        char_loc_flattened = (char_loc[0] * 8 + char_loc[1]).item()
        goal_loc = np.where(state == b"G")
        if len(goal_loc[0]) == 0:
            goal_loc_flattened = char_loc_flattened
        else:
            goal_loc_flattened = (goal_loc[0] * 8 + goal_loc[1]).item()

        next_loc = np.zeros(64)
        if action == 4:
            next_loc[char_loc_flattened] = 1
        elif action == 3:
            if char_loc_flattened - 8 < 0:
                next_loc[char_loc_flattened] = 1
            else:
                next_loc[char_loc_flattened - 8] = 1
        elif action == 2:
            if (char_loc_flattened + 1) % 8 == 0:
                next_loc[char_loc_flattened] = 1
            else:
                next_loc[char_loc_flattened + 1] = 1
        elif action == 1:
            if char_loc_flattened + 8 >= 64:
                next_loc[char_loc_flattened] = 1
            else:
                next_loc[char_loc_flattened + 8] = 1
        elif action == 0:
            if (char_loc_flattened - 1) % 8 == 0:
                next_loc[char_loc_flattened] = 1
            else:
                next_loc[char_loc_flattened - 1] = 1
        else:
            raise ValueError("Action should be [0, ..., 4], got: {}".format(action))

        feature = np.concatenate(
            (
                np.eye(64)[char_loc_flattened],
                next_loc,
                [goal_loc_flattened / 64],
            )
        ).astype(np.float32)

        return feature


class LinearFeature:
    """
    Converts state-action pair into a d-dimensional feature.

    TODO:
    Implement __call__ method which transforms a state-action pair into a d-dimensional vector.
    Here you get to choose your own feature dimensionality
    """

    @property
    def dim(self):
        # TODO: Modify this
        return 64
    
    def __call__(self, state, action):
        """
        Here, a state is representation using a 2d grid of characters.
        For example:
        [
            [b'F' b'F' b'F' b'F' b'H' b'H' b'F' b'F']
            [b'F' b'H' b'H' b'F' b'H' b'F' b'F' b'F']
            [b'H' b'F' b'F' b'F' b'F' b'F' b'F' b'F']
            [b'F' b'F' b'H' b'H' b'F' b'F' b'F' b'F']
            [b'F' b'F' b'F' b'F' b'F' b'H' b'H' b'F']
            [b'F' b'F' b'F' b'F' b'F' b'H' b'F' b'F']
            [b'F' b'H' b'F' b'F' b'H' b'F' b'F' b'F']
            [b'C' b'F' b'S' b'F' b'F' b'G' b'F' b'F']
        ]

        Description:
        - "C" for location of the agent
        - “S” for Start tile
        - “G” for Goal tile
        - “F” for frozen tile
        - “H” for a tile with a hole
        """
        feature = np.zeros(self.dim)

        # ========================================================
        # TODO: Implement linear feature
        # # Get agent position
        # char_loc = np.where(state == b"C")
        # char_loc_flattened = (char_loc[0] * 8 + char_loc[1]).item()

        # # Encode agent's position
        # feature[char_loc_flattened] = 1

        # # One-hot encode the action (5 actions: [0, 1, 2, 3, 4])
        # feature[64 + action] = 1

        # # Encode the surrounding tile types
        # adjacent_tiles = {
        #     0: (-1, 0),  # Left
        #     1: (1, 0),   # Down
        #     2: (0, 1),   # Right
        #     3: (-1, 0),  # Up
        #     4: (0, 0)    # Stay
        # }
        # tile_types = {b'F': 0, b'H': 1, b'S': 2, b'G': 3, b'C': 4}

        # for i, (dx, dy) in adjacent_tiles.items():
        #     new_x, new_y = char_loc[0] + dx, char_loc[1] + dy
        #     if 0 <= new_x < 8 and 0 <= new_y < 8:
        #         tile = state[new_x, new_y]
        #         feature[69 + i] = tile_types.get(tile, 0)  # Default to frozen tile if unknown

        # Find agent position
        char_loc = np.where(state == b"C")
        char_loc_flattened = (char_loc[0] * 8 + char_loc[1]).item()
        feature[char_loc_flattened] = 1  # One-hot encoding for agent position

        # One-hot encoding for action
        feature[64 + action] = 1  

        # Check if goal is reachable
        goal_loc = np.where(state == b"G")
        feature[-1] = 1 if len(goal_loc[0]) > 0 else 0
        # ========================================================

        return feature
