import copy
import itertools
from typing import List, Sequence, Optional, Dict, Union

import gym
import numpy as np
import tree
from minerl.herobraine.hero import spaces


def build_actions(action_space: spaces.Dict,
                  fixed_actions: Dict[str, Union[int, str, np.ndarray]],
                  combinable_actions: Dict[str, List[Union[int, str, np.ndarray]]],
                  not_combinable_actions: Dict[str, List[Union[int, str, np.ndarray]]]):
    # redefine no_op
    no_op = action_space.no_op()
    for action_name, action_value in fixed_actions.items():
        no_op[action_name] = action_value

    def action_from_partial(partial_action: dict) -> dict:
        action = copy.deepcopy(no_op)
        for k, v in partial_action.items():
            action[k] = v
        return action

    actions = []

    for action_values in itertools.product(*list(combinable_actions.values())):
        actions.append(action_from_partial(dict(zip(combinable_actions.keys(), action_values))))

    for action_name, action_values in not_combinable_actions.items():
        for action_value in action_values:
            actions.append(action_from_partial({action_name: action_value}))

    return actions


class MineRLObservationWrapper(object):
    def __init__(self, observation_space):
        super().__init__()
        self._env_observation_space = observation_space
        self._observation_space = gym.spaces.Dict({
            "mainhand_damage": gym.spaces.Box(low=0, high=2304, shape=(1,), dtype=np.int16),
            "mainhand_maxDamage": gym.spaces.Box(low=0, high=2304, shape=(1,), dtype=np.int16),
            "mainhand_type": gym.spaces.Box(low=0, high=2304, shape=(1,), dtype=np.int16),
            "inventory": gym.spaces.Box(
                low=0, high=2304, shape=(len(self._env_observation_space["inventory"].spaces),), dtype=np.int16),
            "pov": gym.spaces.Box(low=0, high=255, shape=(3, 64, 64), dtype=np.uint8)
        })

    def original_space(self):
        return self._env_observation_space

    def transformed_space(self):
        return self._observation_space

    def transform_observation(self, observation):
        mainhand_type = observation['equipped_items.mainhand.type']
        return {
            'mainhand_damage': np.asarray(
                observation['equipped_items.mainhand.damage'], dtype=np.int16).reshape((1,)),
            'mainhand_maxDamage': np.asarray(
                observation['equipped_items.mainhand.maxDamage'], dtype=np.int16).reshape((1,)),
            'mainhand_type': np.asarray(
                self._env_observation_space['equipped_items.mainhand.type'][mainhand_type], dtype=np.int16).reshape((1,)),
            'inventory': np.concatenate([
                np.asarray(v).reshape((1,))
                for _, v in sorted(observation['inventory'].items())], axis=-1).astype(dtype=np.int16),
            'pov': np.transpose(observation['pov'], (2, 0, 1))
        }


class MineRLActionWrapper(object):
    def __init__(self,
                 action_space,
                 fixed_actions: Dict[str, Union[int, str, np.ndarray]],
                 combinable_actions: Dict[str, List[Union[int, str, np.ndarray]]],
                 not_combinable_actions: Dict[str, List[Union[int, str, np.ndarray]]]):
        super().__init__()
        self._env_action_space = action_space
        self._actions = build_actions(
            action_space=self._env_action_space,
            fixed_actions=fixed_actions,
            combinable_actions=combinable_actions,
            not_combinable_actions=not_combinable_actions)
        self._action_space = gym.spaces.Discrete(n=len(self._actions))

    def original_space(self):
        return self._env_action_space

    def transformed_space(self):
        return self._action_space

    def transform_action(self, action):
        def _transform_single_action(a):
            return copy.deepcopy(self._actions[a])
        if isinstance(action, np.ndarray) and action.size > 1:
            actions = [_transform_single_action(int(a)) for a in action]
            return tree.map_structure(lambda *x: np.stack(x), *actions)
        else:
            return _transform_single_action(int(action))
