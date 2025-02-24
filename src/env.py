"""
CSCD84 - Artificial Intelligence, Winter 2025, Assignment 2
B. Chan
"""

from __future__ import annotations

from contextlib import closing
from io import StringIO
from os import path
from typing import Any, SupportsFloat

import numpy as np

import gymnasium as gym
from gymnasium import Env, spaces, utils
from gymnasium.core import ActType, ObsType, RenderFrame
from gymnasium.envs.toy_text.utils import categorical_sample
from gymnasium.error import DependencyNotInstalled
from gymnasium.utils import seeding


LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
STAY = 4


# DFS to check that it's a valid path.
def is_valid(board: list[list[str]], max_size: int) -> bool:
    frontier, discovered = [], set()
    frontier.append((0, 0))
    while frontier:
        r, c = frontier.pop()
        if not (r, c) in discovered:
            discovered.add((r, c))
            directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
            for x, y in directions:
                r_new = r + x
                c_new = c + y
                if r_new < 0 or r_new >= max_size or c_new < 0 or c_new >= max_size:
                    continue
                if board[r_new][c_new] == "G":
                    return True
                if board[r_new][c_new] != "H":
                    frontier.append((r_new, c_new))
    return False


def generate_random_map(size: int = 8, seed: int | None = None) -> list[str]:
    """Generates a random valid map (one that has a path from start to goal)

    Args:
        size: size of each side of the grid
        p: probability that a tile is frozen
        seed: optional seed to ensure the generation of reproducible maps

    Returns:
        A random valid map
    """
    valid = False

    np_random, _ = seeding.np_random(seed)

    while not valid:
        board = np_random.choice(["F", "H"], (size, size), p=(0.8, 0.2))
        idxes = np.array([0, 0, 0, 0])
        while np.all(idxes[:2] == idxes[2:]):
            idxes = np_random.integers(0, size, size=4)
        board[idxes[0]][idxes[1]] = "S"
        board[idxes[2]][idxes[3]] = "G"
        valid = is_valid(board, size)
    return ["".join(x) for x in board], idxes[2:]


class FrozenLakeEnv(Env):
    """
    Modified from:
    1. https://github.com/openai/gym/blob/master/gym/envs/toy_text/frozen_lake.py
    2. https://github.com/chanb/mtil/blob/main/jaxl/envs/toy_text/frozen_lake.py
    """

    metadata = {
        "render_modes": ["human", "ansi", "rgb_array"],
        "render_fps": 20,
    }

    def __init__(
        self,
        horizon: int,
        map_size: int = 8,
        seed: int = None,
        render_mode: str | None = None,
        **kwargs,
    ):
        assert horizon > 0, "horizon must be positive, got: {}".format(horizon)
        self.horizon = horizon

        self._rng = np.random.RandomState(seed)
        desc, goal = generate_random_map(
            size=map_size,
            seed=seed
        )
        slip_to_side = self._rng.uniform(size=(2,))
        while np.sum(slip_to_side) > 1.0:
            slip_to_side = self._rng.uniform(size=(2,))
        self.slip_prob = [
            slip_to_side[0],
            1 - np.sum(slip_to_side),
            slip_to_side[1],
        ]

        self.modified_attributes = {
            "slip_prob": self.slip_prob,
        }

        self.desc = desc = np.asarray(desc, dtype="c")
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.reward_range = (0, 1)

        nA = 5
        nS = nrow * ncol

        self.initial_state_distrib = np.array(desc == b"S").astype("float64").ravel()
        self.initial_state_distrib /= self.initial_state_distrib.sum()

        self.P = {s: {a: [] for a in range(nA)} for s in range(nS)}

        def to_s(row, col):
            return row * ncol + col

        def inc(row, col, a):
            if a == LEFT:
                col = max(col - 1, 0)
            elif a == DOWN:
                row = min(row + 1, nrow - 1)
            elif a == RIGHT:
                col = min(col + 1, ncol - 1)
            elif a == UP:
                row = max(row - 1, 0)
            return (row, col)

        def update_probability_matrix(row, col, action):
            newrow, newcol = inc(row, col, action)
            newstate = to_s(newrow, newcol)
            newletter = desc[newrow, newcol]
            terminated = False
            reward = float(newletter == b"G") - float(newletter == b"H")
            return newstate, reward, terminated

        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(nA):
                    li = self.P[s][a]
                    letter = desc[row, col]
                    if self.slip_prob[1] < 1 and a != STAY:
                        for slip_dir, b in enumerate([(a - 1) % 4, a, (a + 1) % 4]):
                            li.append(
                                (
                                    self.slip_prob[slip_dir],
                                    *update_probability_matrix(row, col, b),
                                )
                            )
                    else:
                        li.append((1.0, *update_probability_matrix(row, col, a)))

        self.observation_space = spaces.Box(low=0, high=1, shape=(nS + 1,))
        self.action_space = spaces.Discrete(nA)

        self.render_mode = render_mode

        # pygame utils
        self.window_size = (min(64 * ncol, 512), min(64 * nrow, 512))
        self.cell_size = (
            self.window_size[0] // self.ncol,
            self.window_size[1] // self.nrow,
        )
        self.window_surface = None
        self.clock = None
        self.hole_img = None
        self.cracked_hole_img = None
        self.ice_img = None
        self.elf_images = None
        self.goal_img = None
        self.start_img = None
        self.n_s = nS
        self.goal = to_s(*goal)

    def get_config(self):
        return {"modified_attributes": self.modified_attributes}

    def get_obs(self):
        state = np.copy(self.desc)
        state[self.s // self.nrow, self.s % self.nrow] = b"C"
        return state

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        transitions = self.P[self.s][int(action)]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, t = transitions[i]
        self.s = s
        self.lastaction = self.lastaction if action == STAY else action

        self.curr_t += 1
        t = self.curr_t >= self.horizon

        if self.render_mode == "human":
            self.render()
        return (self.get_obs(), r, t, False, {"prob": p})

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)
        self.s = categorical_sample(self.initial_state_distrib, self.np_random)
        self.curr_t = 0
        self.lastaction = None

        if self.render_mode == "human":
            self.render()
        return self.get_obs(), {"prob": 1}

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        if self.render_mode == "ansi":
            return self._render_text()
        else:  # self.render_mode in {"human", "rgb_array"}:
            return self._render_gui(self.render_mode)

    def _render_gui(self, mode):
        try:
            import pygame
        except ImportError as e:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[toy-text]`"
            ) from e

        if self.window_surface is None:
            pygame.init()

            if mode == "human":
                pygame.display.init()
                pygame.display.set_caption("Frozen Lake")
                self.window_surface = pygame.display.set_mode(self.window_size)
            elif mode == "rgb_array":
                self.window_surface = pygame.Surface(self.window_size)

        assert (
            self.window_surface is not None
        ), "Something went wrong with pygame. This should never happen."

        if self.clock is None:
            self.clock = pygame.time.Clock()
        if self.hole_img is None:
            file_name = path.join(path.dirname(gym.__file__), "envs/toy_text/img/hole.png")
            self.hole_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.cracked_hole_img is None:
            file_name = path.join(path.dirname(gym.__file__), "envs/toy_text/img/cracked_hole.png")
            self.cracked_hole_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.ice_img is None:
            file_name = path.join(path.dirname(gym.__file__), "envs/toy_text/img/ice.png")
            self.ice_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.goal_img is None:
            file_name = path.join(path.dirname(gym.__file__), "envs/toy_text/img/goal.png")
            self.goal_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.start_img is None:
            file_name = path.join(path.dirname(gym.__file__), "envs/toy_text/img/stool.png")
            self.start_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.elf_images is None:
            elfs = [
                path.join(path.dirname(gym.__file__), "envs/toy_text/img/elf_left.png"),
                path.join(path.dirname(gym.__file__), "envs/toy_text/img/elf_down.png"),
                path.join(path.dirname(gym.__file__), "envs/toy_text/img/elf_right.png"),
                path.join(path.dirname(gym.__file__), "envs/toy_text/img/elf_up.png"),
            ]
            self.elf_images = [
                pygame.transform.scale(pygame.image.load(f_name), self.cell_size)
                for f_name in elfs
            ]

        desc = self.desc.tolist()
        assert isinstance(desc, list), f"desc should be a list or an array, got {desc}"
        for y in range(self.nrow):
            for x in range(self.ncol):
                pos = (x * self.cell_size[0], y * self.cell_size[1])
                rect = (*pos, *self.cell_size)

                self.window_surface.blit(self.ice_img, pos)
                if desc[y][x] == b"H":
                    self.window_surface.blit(self.hole_img, pos)
                elif desc[y][x] == b"G":
                    self.window_surface.blit(self.goal_img, pos)
                elif desc[y][x] == b"S":
                    self.window_surface.blit(self.start_img, pos)

                pygame.draw.rect(self.window_surface, (180, 200, 230), rect, 1)

        # paint the elf
        bot_row, bot_col = self.s // self.ncol, self.s % self.ncol
        cell_rect = (bot_col * self.cell_size[0], bot_row * self.cell_size[1])
        last_action = self.lastaction if self.lastaction is not None else 1
        elf_img = self.elf_images[last_action]

        if desc[bot_row][bot_col] == b"H":
            self.window_surface.blit(self.cracked_hole_img, cell_rect)
        else:
            self.window_surface.blit(elf_img, cell_rect)

        if mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        elif mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window_surface)), axes=(1, 0, 2)
            )

    @staticmethod
    def _center_small_rect(big_rect, small_dims):
        offset_w = (big_rect[2] - small_dims[0]) / 2
        offset_h = (big_rect[3] - small_dims[1]) / 2
        return (
            big_rect[0] + offset_w,
            big_rect[1] + offset_h,
        )

    def _render_text(self):
        desc = self.desc.tolist()
        outfile = StringIO()

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = [[c.decode("utf-8") for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write(f"  ({['Left', 'Down', 'Right', 'Up'][self.lastaction]})\n")
        else:
            outfile.write("\n")
        outfile.write("\n".join("".join(line) for line in desc) + "\n")

        with closing(outfile):
            return outfile.getvalue()

    def close(self) -> None:
        if self.window_surface is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()


# Elf and stool from https://franuka.itch.io/rpg-snow-tileset
# All other assets by Mel Tillery http://www.cyaneus.com/