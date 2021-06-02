import copy
import datetime
import functools
import os
import random
import string
from abc import abstractmethod
from typing import Type, Callable, Union, Sequence, Dict, Optional, NamedTuple, List

import gym
import numpy as np
import torch as th
import tree
from absl import flags, app
from stable_baselines3.common.preprocessing import preprocess_obs
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

import wandb
from agent import LearningAgent, ExtendedEnvWrapper
from minerl_env import MineRLObservationWrapper, MineRLActionWrapper

th.set_num_threads(1)

flags.DEFINE_enum('method', default='rl_rs_mh_sa', enum_values=['rl', 'rl_rs_mh', 'rl_rs_mh_sa'], help='The applied tricks.')

# env
flags.DEFINE_integer('step_multiplier', default=8, help='The pace at which the agent interacts with the environment.')

# train
flags.DEFINE_integer('num_total_train_frames', default=int(2.5e6), help='Number of frames used during training.')
flags.DEFINE_integer('buffer_size', default=100000, help='Replay buffer size of the DQN agents.')
flags.DEFINE_integer('learning_starts', default=10000, help='Number of exploration steps before the training starts.')
flags.DEFINE_integer('target_update_interval', default=5000,
                     help='Interval in number of steps at which the DQN target network is updated.')
flags.DEFINE_string('device', default='cuda', help='The PyTorch device name.')

# eval
flags.DEFINE_integer('eval_episodes', default=200, help='Number of evaluation episodes after training.')

# logging
flags.DEFINE_boolean('wandb_logging', default=False, help='Enables wandb logging.')
flags.DEFINE_string('wandb_project', default='rl-human-prior-tricks', help='Wandb project name.')
flags.DEFINE_string('wandb_entity', default=None, help='Wandb entity (optional).')
flags.DEFINE_list('wandb_tags', default=None, help='Wandb tags (optional).')

FLAGS = flags.FLAGS


class NatureCNN(th.nn.Module):
    def __init__(self, num_input_channels: int = 3):
        super().__init__()
        self._cnn = th.nn.Sequential(
            th.nn.Conv2d(num_input_channels, 32, kernel_size=8, stride=4, padding=0),
            th.nn.ReLU(),
            th.nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            th.nn.ReLU(),
            th.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            th.nn.ReLU(),
            th.nn.Flatten(),
        )

    def forward(self, inputs: th.Tensor) -> th.Tensor:
        return self._cnn(inputs)


class MLP(th.nn.Module):
    def __init__(self, num_input_features: int, hidden_dims: Sequence[int] = (128, 64)):
        super().__init__()
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(th.nn.Linear(num_input_features, hidden_dim))
            layers.append(th.nn.ReLU())
            num_input_features = hidden_dim
        self._mlp = th.nn.Sequential(*layers)

    def forward(self, inputs: th.Tensor) -> th.Tensor:
        return self._mlp(inputs)


class MineRLObtainFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self,
                 observation_space: gym.spaces.Dict,
                 cnn_fn: Type[Callable[[int], th.nn.Module]],
                 fcnn_fn: Type[Callable[[int], th.nn.Module]],
                 features_dim: int = 256):
        super(MineRLObtainFeatureExtractor, self).__init__(observation_space, features_dim=features_dim)

        self._preprocessors = dict()
        for k, v in observation_space.spaces.items():
            if k in ('mainhand_damage', 'mainhand_maxDamage'):
                scale = 1/float(np.max(v.high))
                self._preprocessors[k] = lambda x: x * scale
            elif k == 'inventory':
                self._preprocessors[k] = lambda x: th.log1p(x)
            else:
                self._preprocessors[k] = lambda x: x

        with th.no_grad():
            dummy_obs = {
                k: th.unsqueeze(th.as_tensor(space.sample()), 0)
                for k, space in observation_space.spaces.items()
            }
            dummy_obs = preprocess_obs(obs=dummy_obs, observation_space=observation_space, normalize_images=True)
            spatial_features, scalar_features = self._process_observations(dummy_obs)

        self._cnn = cnn_fn(spatial_features.shape[1])
        self._fcnn = fcnn_fn(scalar_features.shape[1])

        # Compute shape by doing one forward pass
        with th.no_grad():
            spatial_features, scalar_features = self._process_observations(dummy_obs)
            spatial_out = self._cnn(spatial_features)
            scalar_out = self._fcnn(scalar_features)

        self.linear = nn.Sequential(nn.Linear(spatial_out.shape[1] + scalar_out.shape[1], features_dim), nn.ReLU())

    def _process_observations(self, observations):
        observations = tree.map_structure(lambda p, t: p(t), self._preprocessors, observations)

        spatial_features = []
        scalar_features = []
        for k, obs in observations.items():  # tree.flatten(observations):
            if len(obs.shape) == 4:
                spatial_features.append(obs)
            elif len(obs.shape) == 2:
                scalar_features.append(obs)
            else:
                raise ValueError(f"Unexpected observation with shape {obs.shape}")

        spatial_features = th.cat(spatial_features, dim=1)  # th.transpose(th.cat(spatial_features, dim=3), 1, 3)
        scalar_features = th.cat(scalar_features, dim=1)

        return spatial_features, scalar_features

    def forward(self, observations) -> th.Tensor:
        spatial_features, scalar_features = self._process_observations(observations)

        cnn_out = self._cnn(spatial_features)
        fcnn_out = self._fcnn(scalar_features)

        return self.linear(th.cat([cnn_out, fcnn_out], dim=1))


class StepCounterWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.step_counter = 0

    def step(self, action):
        self.step_counter += 1
        return self.env.step(action)


class EpisodeRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.episode_reward = 0

    def reset(self, **kwargs):
        self.episode_reward = 0
        return super().reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.episode_reward += reward
        return obs, reward, done, info


class LoggerWrapper(gym.Wrapper):
    def __init__(self, env, observed_inventory: Optional[Sequence[str]] = None):
        super().__init__(env)
        self._observed_inventory = observed_inventory or [
            'coal', 'cobblestone', 'crafting_table', 'furnace', 'iron_ingot', 'iron_ore', 'iron_pickaxe', 'log',
            'planks', 'stick', 'stone', 'stone_pickaxe', 'wooden_pickaxe']
        self._prev_inventory = None  # {key: 0 for key in self._observed_inventory}

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        self._prev_inventory = {key: obs['inventory'][key] for key in self._observed_inventory}
        return obs

    def _report_changes(self, obs, reward, done, info):
        changes = []
        if reward != 0:
            changes.append(('reward', reward))
        if done:
            changes.append(('done', done))
        for key in self._observed_inventory:
            if self._prev_inventory[key] != obs['inventory'][key]:
                changes.append((f'inventory/{key}', self._prev_inventory[key], obs['inventory'][key]))
                self._prev_inventory[key] = obs['inventory'][key]
        if len(changes) > 0:
            print(f"Env logger | " + ", ".join([f"{c[0]}: {c[1] if len(c) == 2 else f'{c[1]} -> {c[2]}'}" for c in changes]))

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._report_changes(obs, reward, done, info)
        return obs, reward, done, info


class MineRLFrameSkipWrapper(gym.Wrapper):
    def __init__(self, env, step_size: int):
        super().__init__(env)
        assert step_size > 0
        self.step_size = step_size
        self.total_frames = 0

    def step(self, action: dict):
        obs, total_reward, done, info = self.env.step(action)
        self.total_frames += 1
        action = copy.deepcopy(action)
        # the following action parameters should not be repeated:
        action['craft'] = 'none'
        action['nearbyCraft'] = 'none'
        action['nearbySmelt'] = 'none'
        action['place'] = 'none'
        for _ in range(self.step_size - 1):
            if done:
                break
            obs, reward, done, info = self.env.step(action)
            self.total_frames += 1
            total_reward += reward
        return obs, total_reward, done, info


def check_inventory(env, required_amounts: dict) -> bool:
    inventory = env.get_current_observation()['inventory']
    for k, v in required_amounts.items():
        if inventory[k] < v:
            print(f"Inventory check failed: {k}={inventory[k]} < {v}")
            return False
    return True


class MineRLAgent(object):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def check_precondition(self, env) -> bool:
        pass


class CollectAgent(LearningAgent, MineRLAgent):
    def __init__(self,
                 item_name: str,
                 goal_amount: int,
                 inventory_precondition: Dict[str, int],
                 minerl_observation_wrapper: MineRLObservationWrapper,
                 minerl_action_wrapper: MineRLActionWrapper,
                 env_observation_space: gym.spaces.Space,
                 env_action_space: gym.spaces.Space,
                 batch_size: int = 64,
                 exploration_steps: int = 10000):
        super().__init__(env_observation_space, env_action_space, batch_size, exploration_steps)
        self.item_name = item_name
        self.item_index = list(sorted(self.env_observation_space["inventory"].spaces.keys())).index(self.item_name)
        self.goal_amount = goal_amount
        self.inventory_precondition = inventory_precondition
        self.minerl_observation_wrapper = minerl_observation_wrapper
        self.minerl_action_wrapper = minerl_action_wrapper
        self.agent_observation_space = minerl_observation_wrapper.transformed_space()
        self.agent_action_space = minerl_action_wrapper.transformed_space()
        self.rotation = np.asarray([0, 0])  # keep track of agent camera rotation

    @property
    def name(self) -> str:
        return f"{self.item_name}_agent"

    def check_precondition(self, env) -> bool:
        return check_inventory(env, self.inventory_precondition)

    def env_observation_to_agent(self, obs: Union[Dict[str, np.ndarray], np.ndarray]) -> Union[Dict[str, np.ndarray], np.ndarray]:
        return self.minerl_observation_wrapper.transform_observation(obs)

    def agent_action_to_env(self, action: np.ndarray):
        action = self.minerl_action_wrapper.transform_action(action)
        self.rotation = self.rotation + action['camera'] * FLAGS.step_multiplier
        self.rotation = ((self.rotation + 180) % 360) - 180
        return action

    def env_reward_to_agent(self, agent_prev_obs, agent_obs, reward, env_done, info) -> float:
        return max(0, (agent_obs['inventory'][self.item_index] - agent_prev_obs['inventory'][self.item_index]).squeeze())

    def env_done_to_agent(self, agent_prev_obs, agent_obs, reward, done, info) -> bool:
        return done or agent_obs['inventory'][self.item_index].squeeze() >= self.goal_amount


class CollectMultiAgent(LearningAgent, MineRLAgent):
    def __init__(self,
                 items: Dict[str, int],
                 inventory_precondition: Dict[str, int],
                 minerl_observation_wrapper: MineRLObservationWrapper,
                 minerl_action_wrapper: MineRLActionWrapper,
                 env_observation_space: gym.spaces.Space,
                 env_action_space: gym.spaces.Space,
                 batch_size: int = 64,
                 exploration_steps: int = 10000):
        super().__init__(env_observation_space, env_action_space, batch_size, exploration_steps)
        self.items = items
        self.item_index = {
            k: list(sorted(self.env_observation_space["inventory"].spaces.keys())).index(k) for k in items.keys()
        }
        self.inventory_precondition = inventory_precondition
        self.minerl_observation_wrapper = minerl_observation_wrapper
        self.minerl_action_wrapper = minerl_action_wrapper
        self.agent_observation_space = minerl_observation_wrapper.transformed_space()
        self.agent_action_space = minerl_action_wrapper.transformed_space()
        self.rotation = np.asarray([0, 0])  # keep track of agent camera rotation

    @property
    def name(self) -> str:
        return f"{'_'.join(list(sorted(self.items.keys())))}_agent"

    def check_precondition(self, env) -> bool:
        return check_inventory(env, self.inventory_precondition)

    def env_observation_to_agent(self, obs: Union[Dict[str, np.ndarray], np.ndarray]) -> Union[Dict[str, np.ndarray], np.ndarray]:
        return self.minerl_observation_wrapper.transform_observation(obs)

    def agent_action_to_env(self, action: np.ndarray):
        action = self.minerl_action_wrapper.transform_action(action)
        self.rotation = self.rotation + action['camera'] * FLAGS.step_multiplier
        self.rotation = ((self.rotation + 180) % 360) - 180
        return action

    def env_reward_to_agent(self, agent_prev_obs, agent_obs, reward, env_done, info) -> float:
        reward = 0
        for k, v in self.items.items():
            reward += max(0, (min(agent_obs['inventory'][self.item_index[k]], v) - agent_prev_obs['inventory'][self.item_index[k]]).squeeze())
        return reward

    def env_done_to_agent(self, agent_prev_obs, agent_obs, reward, done, info) -> bool:
        return done or all(agent_obs['inventory'][self.item_index[k]].squeeze() >= v for k, v in self.items.items())


class CollectDiamondAgent(LearningAgent, MineRLAgent):
    def __init__(self,
                 minerl_observation_wrapper: MineRLObservationWrapper,
                 minerl_action_wrapper: MineRLActionWrapper,
                 env_observation_space: gym.spaces.Space,
                 env_action_space: gym.spaces.Space,
                 batch_size: int = 64,
                 exploration_steps: int = 10000):
        super().__init__(env_observation_space, env_action_space, batch_size, exploration_steps)
        self.minerl_observation_wrapper = minerl_observation_wrapper
        self.minerl_action_wrapper = minerl_action_wrapper
        self.agent_observation_space = minerl_observation_wrapper.transformed_space()
        self.agent_action_space = minerl_action_wrapper.transformed_space()

    @property
    def name(self) -> str:
        return "diamond_agent"

    def check_precondition(self, env) -> bool:
        return check_inventory(env, {'iron_pickaxe': 1})

    def env_observation_to_agent(self, obs: Union[Dict[str, np.ndarray], np.ndarray]) -> Union[Dict[str, np.ndarray], np.ndarray]:
        return self.minerl_observation_wrapper.transform_observation(obs)

    def agent_action_to_env(self, action: np.ndarray):
        return self.minerl_action_wrapper.transform_action(action)

    def env_reward_to_agent(self, agent_prev_obs, agent_obs, reward, env_done, info) -> float:
        return 1. if reward == 1024 else 0.

    def env_done_to_agent(self, agent_prev_obs, agent_obs, reward, done, info) -> bool:
        return done


class DefaultRewardAgent(LearningAgent, MineRLAgent):
    def __init__(self,
                 minerl_observation_wrapper: MineRLObservationWrapper,
                 minerl_action_wrapper: MineRLActionWrapper,
                 env_observation_space: gym.spaces.Space,
                 env_action_space: gym.spaces.Space,
                 batch_size: int = 64,
                 exploration_steps: int = 10000):
        super().__init__(env_observation_space, env_action_space, batch_size, exploration_steps)
        self.minerl_observation_wrapper = minerl_observation_wrapper
        self.minerl_action_wrapper = minerl_action_wrapper
        self.agent_observation_space = minerl_observation_wrapper.transformed_space()
        self.agent_action_space = minerl_action_wrapper.transformed_space()

    @property
    def name(self) -> str:
        return "default_reward_agent"

    def check_precondition(self, env) -> bool:
        return True

    def env_observation_to_agent(self, obs: Union[Dict[str, np.ndarray], np.ndarray]) -> Union[Dict[str, np.ndarray], np.ndarray]:
        return self.minerl_observation_wrapper.transform_observation(obs)

    def agent_action_to_env(self, action: np.ndarray):
        return self.minerl_action_wrapper.transform_action(action)


class ScriptedAgent(MineRLAgent):
    def __init__(self, actions: list, inventory_precondition: Dict[str, int]):
        self.actions = actions
        self.inventory_precondition = inventory_precondition

    @property
    def name(self) -> str:
        return "scripted_agent"

    def check_precondition(self, env) -> bool:
        return check_inventory(env, self.inventory_precondition)

    def play(self,
             env: ExtendedEnvWrapper,
             termination_callback: Optional[Callable[[ExtendedEnvWrapper, ], bool]] = None,
             **kwargs):
        step = 0
        while not env.needs_reset() \
                and (termination_callback is None or not termination_callback(env)) \
                and step < len(self.actions):
            _ = env.step(self.actions[step])
            step += 1



def from_partial_action(env, partial_action: dict):
    action = env.action_space.no_op()
    for k, v in partial_action.items():
        action[k] = v
    return action


class EpisodeOutcome(NamedTuple):
    reward: float
    hierarchy: int


class SequentialControllerAgent(object):

    def __init__(self, sub_agents: List[Union[LearningAgent, ScriptedAgent]]) -> None:
        super().__init__()
        self.sub_agents = sub_agents

    def play(self,
             env: ExtendedEnvWrapper,
             training: bool = True,
             eval_epsilon=0.1,
             **kwargs):
        for agent_index, agent in enumerate(self.sub_agents):
            if env.needs_reset() or not agent.check_precondition(env):
                return EpisodeOutcome(reward=env.episode_reward, hierarchy=agent_index)
            agent.play(env, training=training, eval_epsilon=eval_epsilon)
        return EpisodeOutcome(reward=env.episode_reward, hierarchy=len(self.sub_agents))


def rl_agent(env, dqn_arguments):
    diamond_agent = DefaultRewardAgent(
        minerl_observation_wrapper=MineRLObservationWrapper(
            observation_space=env.observation_space),
        minerl_action_wrapper=MineRLActionWrapper(
            action_space=env.action_space,
            fixed_actions={"attack": 1},
            combinable_actions={
                "forward": [0, 1],
                "jump": [0, 1],
                "equip": ['none', 'wooden_pickaxe', 'stone_pickaxe', 'iron_pickaxe'],
            },
            not_combinable_actions={
                "camera": [np.asarray([1, 0]), np.asarray([-1, 0]), np.asarray([0, 1]), np.asarray([0, -1])],
                "craft": ['crafting_table', 'planks', 'stick'],
                "nearbyCraft": ['furnace', 'wooden_pickaxe', 'stone_pickaxe', 'iron_pickaxe'],
                "nearbySmelt": ['iron_ingot'],
                "place": ['crafting_table', 'furnace'],
            }),
        env_observation_space=env.observation_space,
        env_action_space=env.action_space,
        exploration_steps=10000)
    diamond_agent.initialize_rl(dqn_arguments=dqn_arguments, use_cnn=True)
    return SequentialControllerAgent(sub_agents=[diamond_agent])


def rl_rs_mh_agent(env, dqn_arguments):
    log_agent = CollectAgent(
        item_name='log',
        goal_amount=4,
        inventory_precondition={},
        minerl_observation_wrapper=MineRLObservationWrapper(observation_space=env.observation_space),
        minerl_action_wrapper=MineRLActionWrapper(
            action_space=env.action_space,
            fixed_actions={"attack": 1},
            combinable_actions={"forward": [0, 1], "jump": [0, 1]},
            not_combinable_actions={"camera": [np.asarray([0, 1]), np.asarray([0, -1])]}),
        env_observation_space=env.observation_space,
        env_action_space=env.action_space,
        exploration_steps=10000)
    log_agent.initialize_rl(dqn_arguments=dqn_arguments, use_cnn=True)

    planks_agent = CollectAgent(
        item_name='planks',
        goal_amount=12,
        inventory_precondition={'log': 4},
        minerl_observation_wrapper=MineRLObservationWrapper(observation_space=env.observation_space),
        minerl_action_wrapper=MineRLActionWrapper(
            action_space=env.action_space,
            fixed_actions={},
            combinable_actions={},
            not_combinable_actions={'craft': ['none', 'planks']}),
        env_observation_space=env.observation_space,
        env_action_space=env.action_space,
        exploration_steps=10000)
    planks_agent.initialize_rl(dqn_arguments=dqn_arguments, use_cnn=True)

    stick_crafting_table_agent = CollectMultiAgent(
        items={
            'stick': 8,
            'crafting_table': 1
        },
        inventory_precondition={'planks': 8},
        minerl_observation_wrapper=MineRLObservationWrapper(observation_space=env.observation_space),
        minerl_action_wrapper=MineRLActionWrapper(
            action_space=env.action_space,
            fixed_actions={},
            combinable_actions={},
            not_combinable_actions={'craft': ['none', 'stick', 'crafting_table']}),
        env_observation_space=env.observation_space,
        env_action_space=env.action_space,
        exploration_steps=10000)
    stick_crafting_table_agent.initialize_rl(dqn_arguments=dqn_arguments, use_cnn=True)

    wooden_pickaxe_agent = CollectAgent(
        item_name='wooden_pickaxe',
        goal_amount=1,
        inventory_precondition={'crafting_table': 1, 'stick': 2, 'planks': 3},
        minerl_observation_wrapper=MineRLObservationWrapper(observation_space=env.observation_space),
        minerl_action_wrapper=MineRLActionWrapper(
            action_space=env.action_space,
            fixed_actions={},
            combinable_actions={"forward": [0, 1], "attack": [0, 1]},
            not_combinable_actions={
                "camera": [np.asarray([1, 0]), np.asarray([-1, 0]), np.asarray([0, 1]), np.asarray([0, -1])],
                'place': ['crafting_table'],
                'nearbyCraft': ['wooden_pickaxe'],
            }),
        env_observation_space=env.observation_space,
        env_action_space=env.action_space,
        exploration_steps=10000)
    wooden_pickaxe_agent.initialize_rl(dqn_arguments=dqn_arguments, use_cnn=True)

    collect_crafting_table_agent = CollectAgent(
        item_name='crafting_table',
        goal_amount=1,
        inventory_precondition={},
        minerl_observation_wrapper=MineRLObservationWrapper(observation_space=env.observation_space),
        minerl_action_wrapper=MineRLActionWrapper(
            action_space=env.action_space,
            fixed_actions={"attack": 1},
            combinable_actions={},
            not_combinable_actions={"camera": [np.asarray([0, 0]), np.asarray([0, 1]), np.asarray([0, -1])]}),
        env_observation_space=env.observation_space,
        env_action_space=env.action_space,
        exploration_steps=10000)
    collect_crafting_table_agent.initialize_rl(dqn_arguments=dqn_arguments, use_cnn=True)

    cobblestone_agent = CollectAgent(
        item_name='cobblestone',
        goal_amount=11,
        inventory_precondition={},
        minerl_observation_wrapper=MineRLObservationWrapper(observation_space=env.observation_space),
        minerl_action_wrapper=MineRLActionWrapper(
            action_space=env.action_space,
            fixed_actions={"attack": 1, "forward": 1, "equip": "wooden_pickaxe"},
            combinable_actions={"jump": [0, 1]},
            not_combinable_actions={
                "camera": [np.asarray([1, 0]), np.asarray([-1, 0]), np.asarray([0, 1]), np.asarray([0, -1])],
            }),
        env_observation_space=env.observation_space,
        env_action_space=env.action_space,
        exploration_steps=10000)
    cobblestone_agent.initialize_rl(dqn_arguments=dqn_arguments, use_cnn=True)

    stone_pickaxe_furnace_agent = CollectMultiAgent(
        items={
            'stone_pickaxe': 1,
            'furnace': 1
        },
        inventory_precondition={'cobblestone': 11},
        minerl_observation_wrapper=MineRLObservationWrapper(observation_space=env.observation_space),
        minerl_action_wrapper=MineRLActionWrapper(
            action_space=env.action_space,
            fixed_actions={},
            combinable_actions={"forward": [0, 1], "attack": [0, 1]},
            not_combinable_actions={
                "camera": [np.asarray([1, 0]), np.asarray([-1, 0]), np.asarray([0, 1]), np.asarray([0, -1])],
                'place': ['crafting_table'],
                'nearbyCraft': ['stone_pickaxe', 'furnace'],
            }),
        env_observation_space=env.observation_space,
        env_action_space=env.action_space,
        exploration_steps=10000)
    stone_pickaxe_furnace_agent.initialize_rl(dqn_arguments=dqn_arguments, use_cnn=True)

    iron_ore_agent = CollectAgent(
        item_name='iron_ore',
        goal_amount=3,
        inventory_precondition={
            'stone_pickaxe': 1,
        },
        minerl_observation_wrapper=MineRLObservationWrapper(
            observation_space=env.observation_space),
        minerl_action_wrapper=MineRLActionWrapper(
            action_space=env.action_space,
            fixed_actions={"attack": 1, "forward": 1, "equip": "stone_pickaxe"},
            combinable_actions={"jump": [0, 1]},
            not_combinable_actions={
                "camera": [np.asarray([1, 0]), np.asarray([-1, 0]), np.asarray([0, 1]), np.asarray([0, -1])]
            }),
        env_observation_space=env.observation_space,
        env_action_space=env.action_space,
        exploration_steps=10000)
    iron_ore_agent.initialize_rl(dqn_arguments=dqn_arguments, use_cnn=True)

    iron_ingot_agent = CollectAgent(
        item_name='iron_ingot',
        goal_amount=3,
        inventory_precondition={'furnace': 1, 'iron_ore': 3},
        minerl_observation_wrapper=MineRLObservationWrapper(observation_space=env.observation_space),
        minerl_action_wrapper=MineRLActionWrapper(
            action_space=env.action_space,
            fixed_actions={},
            combinable_actions={"forward": [0, 1], "attack": [0, 1]},
            not_combinable_actions={
                "camera": [np.asarray([1, 0]), np.asarray([-1, 0]), np.asarray([0, 1]), np.asarray([0, -1])],
                'place': ['furnace'],
                'nearbySmelt': ['iron_ingot'],
            }),
        env_observation_space=env.observation_space,
        env_action_space=env.action_space,
        exploration_steps=10000)
    iron_ingot_agent.initialize_rl(dqn_arguments=dqn_arguments, use_cnn=True)

    iron_pickaxe_agent = CollectAgent(
        item_name='iron_pickaxe',
        goal_amount=1,
        inventory_precondition={'iron_ingot': 3, 'stick': 2, 'crafting_table': 1},
        minerl_observation_wrapper=MineRLObservationWrapper(observation_space=env.observation_space),
        minerl_action_wrapper=MineRLActionWrapper(
            action_space=env.action_space,
            fixed_actions={},
            combinable_actions={"forward": [0, 1], "attack": [0, 1]},
            not_combinable_actions={
                "camera": [np.asarray([1, 0]), np.asarray([-1, 0]), np.asarray([0, 1]), np.asarray([0, -1])],
                'place': ['crafting_table'],
                'nearbyCraft': ['iron_pickaxe'],
            }),
        env_observation_space=env.observation_space,
        env_action_space=env.action_space,
        exploration_steps=10000)
    iron_pickaxe_agent.initialize_rl(dqn_arguments=dqn_arguments, use_cnn=True)

    diamond_agent = CollectDiamondAgent(
        minerl_observation_wrapper=MineRLObservationWrapper(
            observation_space=env.observation_space),
        minerl_action_wrapper=MineRLActionWrapper(
            action_space=env.action_space,
            fixed_actions={"attack": 1, "forward": 1, "equip": "iron_pickaxe"},
            combinable_actions={"jump": [0, 1]},
            not_combinable_actions={
                "camera": [np.asarray([1, 0]), np.asarray([-1, 0]), np.asarray([0, 1]), np.asarray([0, -1])]
            }),
        env_observation_space=env.observation_space,
        env_action_space=env.action_space,
        exploration_steps=10000)
    diamond_agent.initialize_rl(dqn_arguments=dqn_arguments, use_cnn=True)

    return SequentialControllerAgent(sub_agents=[
        log_agent,
        planks_agent,
        stick_crafting_table_agent,
        wooden_pickaxe_agent,
        collect_crafting_table_agent,
        cobblestone_agent,
        stone_pickaxe_furnace_agent,
        collect_crafting_table_agent,
        iron_ore_agent,
        iron_ingot_agent,
        iron_pickaxe_agent,
        diamond_agent
    ])


def rl_rs_mh_sa_agent(env, dqn_arguments):
    make_action = functools.partial(from_partial_action, env)

    log_agent = CollectAgent(
        item_name='log',
        goal_amount=4,
        inventory_precondition={},
        minerl_observation_wrapper=MineRLObservationWrapper(
            observation_space=env.observation_space),
        minerl_action_wrapper=MineRLActionWrapper(
            action_space=env.action_space,
            fixed_actions={"attack": 1},
            combinable_actions={"forward": [0, 1], "jump": [0, 1]},
            not_combinable_actions={"camera": [np.asarray([0, 1]), np.asarray([0, -1])]}),
        env_observation_space=env.observation_space,
        env_action_space=env.action_space,
        exploration_steps=10000)
    log_agent.initialize_rl(dqn_arguments=dqn_arguments, use_cnn=True)

    # todo: needs workaround if the agent is in water O:-)
    wooden_pickaxe_agent = ScriptedAgent(
        actions=(
            [make_action({'craft': 'planks'})] * 4 +            # craft planks four times
            [make_action({'craft': 'stick'})] * 2 +             # craft sticks two times
            [make_action({'craft': 'crafting_table'})] +        # craft crafting table
            [make_action({'camera': np.asarray([90, 0])})] +    # look down
            [make_action({'attack': 1})] * 10 +                 # get rid of block below
            [make_action({'forward': 1})] +                     # jitter around to make sure the agent falls into the hole
            [make_action({'left': 1})] +                        #
            [make_action({'back': 1})] +                        #
            [make_action({'right': 1})] +                       #
            [make_action({'right': 1})] +                       #
            [make_action({'back': 1})] +                        #
            [make_action({'left': 1})] +                        #
            [make_action({'forward': 1})] +                     #
            [make_action({'jump': 1})] +                        # jump to make space for the crafting table
            [make_action({'place': 'crafting_table'})] +        # place crafting table
            [make_action({'camera': np.asarray([-1, 0])})] +    # look up 1 degree
            [make_action({})] * 2 +                             # wait for crafting table
            [make_action({'nearbyCraft': 'wooden_pickaxe'})] +  # craft wooden pickaxe
            [make_action({'camera': np.asarray([1, 0])})] +     # look down 1 degree
            [make_action({'attack': 1})] * 25 +                 # collect crafting table
            [make_action({'camera': np.asarray([-90, 0])})]     # look back up 90 degrees
        ),
        inventory_precondition={
            'log': 4
        }
    )

    dig_down_agent = ScriptedAgent(
        actions=(
            [make_action({'camera': np.asarray([90, 0])})] +                    # look down 90 degrees
            [make_action({'attack': 1, 'equip': 'wooden_pickaxe'})] * 150 +     # dig down for 60 seconds
            [make_action({'camera': np.asarray([-60, 0])})]                     # look up 60 degrees
        ),
        inventory_precondition={
            'wooden_pickaxe': 1
        }
    )

    # stone_camera_actions = [np.asarray([1, 0]), np.asarray([-1, 0]), np.asarray([0, 1]), np.asarray([0, -1])]
    stone_camera_actions = [np.asarray([0, 1]), np.asarray([0, -1])]
    stone_agent = CollectAgent(
        item_name='cobblestone',
        goal_amount=11,
        inventory_precondition={
            'wooden_pickaxe': 1
        },
        minerl_observation_wrapper=MineRLObservationWrapper(
            observation_space=env.observation_space),
        minerl_action_wrapper=MineRLActionWrapper(
            action_space=env.action_space,
            fixed_actions={"attack": 1, "forward": 1, "equip": "wooden_pickaxe"},
            combinable_actions={"jump": [0, 1]},
            not_combinable_actions={"camera": stone_camera_actions}),
        env_observation_space=env.observation_space,
        env_action_space=env.action_space,
        exploration_steps=10000)
    stone_agent.initialize_rl(dqn_arguments=dqn_arguments, use_cnn=True)

    stone_pickaxe_furnace_agent = ScriptedAgent(
        actions=(
            [make_action({'camera': np.asarray([60, 0])})] +      # look down
            [make_action({'attack': 1})] * 25 +                   # get rid of block below
            [make_action({'jump': 1})] +                          # jump to make space for the crafting table
            [make_action({'place': 'crafting_table'})] +          # place crafting table
            [make_action({'camera': np.asarray([-1, 0])})] +      # look up 1 degree
            [make_action({})] * 2 +                               # wait for crafting table
            [make_action({'nearbyCraft': 'stone_pickaxe'})] +     # craft wooden pickaxe
            [make_action({'nearbyCraft': 'furnace'})] +           # craft wooden pickaxe
            [make_action({'camera': np.asarray([1, 0])})] +       # look down 1 degree
            [make_action({'attack': 1})] * 25 +                   # collect crafting table
            [make_action({'camera': np.asarray([-60, 0])})]       # look back up 90 degrees
        ),
        inventory_precondition={
            'cobblestone': 11,
            'stick': 2,
            'crafting_table': 1,
        }
    )

    iron_ore_camera_actions = [np.asarray([1, 0]), np.asarray([-1, 0]), np.asarray([0, 1]), np.asarray([0, -1])]
    iron_ore_agent = CollectAgent(
        item_name='iron_ore',
        goal_amount=3,
        inventory_precondition={
            'stone_pickaxe': 1,
        },
        minerl_observation_wrapper=MineRLObservationWrapper(
            observation_space=env.observation_space),
        minerl_action_wrapper=MineRLActionWrapper(
            action_space=env.action_space,
            fixed_actions={"attack": 1, "forward": 1, "equip": "stone_pickaxe"},
            combinable_actions={"jump": [0, 1]},
            not_combinable_actions={"camera": iron_ore_camera_actions}),
        env_observation_space=env.observation_space,
        env_action_space=env.action_space,
        exploration_steps=10000)
    iron_ore_agent.initialize_rl(dqn_arguments=dqn_arguments, use_cnn=True)

    iron_ingot_agent = ScriptedAgent(
        actions=(
            # note camera rotation is messed up this time, let's try to step back and place furnace
            [make_action({'jump': 1, 'back': 1})] * 2 +         # step back
            [make_action({'place': 'furnace'})] +               # place furnace
            [make_action({})] * 2 +                             # wait for furnace
            [make_action({'nearbySmelt': 'iron_ingot'})] * 3    # smelt iron ingot
        ),
        inventory_precondition={
            'iron_ore': 3,
            'furnace': 1,
        }
    )

    iron_pickaxe_agent = ScriptedAgent(
        actions=(
            # note camera rotation is messed up this time, let's try to step back and place crafting_table
            [make_action({'jump': 1, 'back': 1})] * 2 +     # step back
            [make_action({'place': 'crafting_table'})] +    # place crafting_table
            [make_action({})] * 2 +                         # wait for crafting_table
            [make_action({'nearbyCraft': 'iron_pickaxe'})]  # craft iron pickaxe
        ),
        inventory_precondition={
            'iron_ingot': 3,
            'stick': 2,
            'crafting_table': 1,
        }
    )

    diamond_camera_actions = [np.asarray([1, 0]), np.asarray([-1, 0]), np.asarray([0, 1]), np.asarray([0, -1])]
    diamond_agent = CollectDiamondAgent(
        minerl_observation_wrapper=MineRLObservationWrapper(
            observation_space=env.observation_space),
        minerl_action_wrapper=MineRLActionWrapper(
            action_space=env.action_space,
            fixed_actions={"attack": 1, "forward": 1, "equip": "iron_pickaxe"},
            combinable_actions={"jump": [0, 1]},
            not_combinable_actions={"camera": diamond_camera_actions}),
        env_observation_space=env.observation_space,
        env_action_space=env.action_space,
        exploration_steps=10000)
    diamond_agent.initialize_rl(dqn_arguments=dqn_arguments, use_cnn=True)

    return SequentialControllerAgent(sub_agents=[
        log_agent,
        wooden_pickaxe_agent,
        dig_down_agent,
        stone_agent,
        stone_pickaxe_furnace_agent,
        iron_ore_agent,
        iron_ingot_agent,
        iron_pickaxe_agent,
        diamond_agent
    ])


def main(_):
    run_name = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-" \
               f"{''.join(random.choices(string.ascii_lowercase, k=5))}"

    dqn_arguments = {
        'buffer_size': FLAGS.buffer_size,
        'learning_starts': FLAGS.learning_starts,
        'target_update_interval': FLAGS.target_update_interval,
        'device': FLAGS.device,
        'policy_kwargs': {
            'features_extractor_class': MineRLObtainFeatureExtractor,
            'features_extractor_kwargs': {
                'cnn_fn': lambda num_features: NatureCNN(num_features),
                'fcnn_fn': lambda num_features: MLP(num_features, hidden_dims=(128, 64)),
                'features_dim': 256
            },
            'normalize_images': True
        }
    }

    if FLAGS.wandb_logging:
        wandb.init(
            project=FLAGS.wandb_project,
            entity=FLAGS.wandb_entity,
            tags=FLAGS.wandb_tags,
            config={
                'flags': FLAGS.flag_values_dict(),
                'dqn_arguments': dqn_arguments,
            },
        )

    env = gym.make('MineRLObtainDiamond-v0')
    env = StepCounterWrapper(env)
    env = MineRLFrameSkipWrapper(env, FLAGS.step_multiplier)
    env = EpisodeRewardWrapper(env)
    env = LoggerWrapper(env)
    env = ExtendedEnvWrapper(env)

    if FLAGS.method == 'rl':
        agent = rl_agent(env, dqn_arguments)
    elif FLAGS.method == 'rl_rs_mh':
        agent = rl_rs_mh_agent(env, dqn_arguments)
    elif FLAGS.method == 'rl_rs_mh_sa':
        agent = rl_rs_mh_sa_agent(env, dqn_arguments)
    else:
        raise ValueError(f"Unknown method: {FLAGS.method}")

    def log_agent_episode(agent, episode: int):
        if isinstance(agent, SequentialControllerAgent):
            for agent in set(agent.sub_agents):
                log_agent_episode(agent, episode)
        elif isinstance(agent, (CollectAgent, CollectDiamondAgent, DefaultRewardAgent)):
            print(f'{agent.name}: '
                  f'updates={agent.num_network_updates},  '
                  f'buffer_size={agent.dqn_agent.replay_buffer.size()},  '
                  f'eps={1 - min(1. - 0.1, agent.num_network_updates / agent.exploration_steps)}')
            if FLAGS.wandb_logging:
                wandb.log({
                    f'{agent.name}/updates': agent.num_network_updates,
                    f'{agent.name}/buffer_size': agent.dqn_agent.replay_buffer.size(),
                    f'{agent.name}/eps': 1 - min(1. - 0.1, agent.num_network_updates / agent.exploration_steps),
                }, episode)

    def log_episode(values: dict, episode: int):
        print(f"Episode {episode}: {', '.join([f'{k}={v}' for k, v in values.items()])}", flush=True)
        log_file_path = f'minerl-runs/{FLAGS.method}/{run_name}.train'
        if not os.path.exists(log_file_path):
            os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
            with open(log_file_path, mode='w') as f:
                f.write('{0}\n'.format('\t'.join(['', 'method', 'num_frames', 'episode_reward'])))
        with open(log_file_path, mode='a') as f:
            f.write('{0}\n'.format(
                '\t'.join([str(episode), FLAGS.method, str(values['num_frames']), str(values['episode_reward'])])))
        if FLAGS.wandb_logging:
            wandb.log(values, episode)

    def log_eval_outcome(values: dict, episode: int):
        print(f"Eval outcome: {', '.join([f'{k}={v}' for k, v in values.items()])}", flush=True)
        log_file_path = f'minerl-runs/{FLAGS.method}/{run_name}.eval'
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        with open(log_file_path, mode='w') as f:
            f.write('{0}\n'.format('\t'.join(['', 'episode_reward', 'hierarchy'])))
        with open(log_file_path, mode='a') as f:
            for i, (episode_reward, hierarchy) in enumerate(
                    zip(values['eval/episode_rewards'], values['eval/hierarchies'])):
                f.write('{0}\n'.format('\t'.join([str(i), str(episode_reward), str(hierarchy)])))
        if FLAGS.wandb_logging:
            wandb.log(values, episode)

    train_episodes = 0
    while env.total_frames < FLAGS.num_total_train_frames:
        _ = env.reset()
        episode_outcome = agent.play(env=env, training=True)
        log_episode({
            'episode_reward': episode_outcome.reward,
            'hierarchy': episode_outcome.hierarchy,
            'num_frames': env.total_frames,
            'episode': train_episodes
        }, train_episodes)
        log_agent_episode(agent, episode=train_episodes)
        train_episodes += 1

    eval_outcomes = []
    for i in range(FLAGS.eval_episodes):
        _ = env.reset()
        episode_outcome = agent.play(env=env, training=False, eval_epsilon=0.1)
        eval_outcomes.append(episode_outcome)
        print(f"Eval episode {i}: "
              f"reward={episode_outcome.reward} (mean={np.mean([o.reward for o in eval_outcomes])}), "
              f"hierarchy={episode_outcome.hierarchy} (mean={np.mean([o.hierarchy for o in eval_outcomes])})",
              flush=True)

    eval_outcome = tree.map_structure(lambda *x: np.stack(x), *eval_outcomes)

    log_eval_outcome({
        'eval/episode_rewards': eval_outcome.reward,
        'eval/hierarchies': eval_outcome.hierarchy,
        'eval/episode_reward/mean': eval_outcome.reward.mean(),
        'eval/hierarchy/mean': eval_outcome.hierarchy.mean(),
        'eval/episode_reward/min': eval_outcome.reward.min(),
        'eval/hierarchy/min': eval_outcome.hierarchy.min(),
        'eval/episode_reward/max': eval_outcome.reward.max(),
        'eval/hierarchy/max': eval_outcome.hierarchy.max()
    }, train_episodes)


if __name__ == "__main__":
    os.environ['OMP_NUM_THREADS '] = '1'
    app.run(main)
