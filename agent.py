# Main class of the learning agent
from typing import Optional, Callable
from collections import deque

import numpy as np
import gym

from stable_baselines3 import DQN
from stable_baselines3.common.utils import polyak_update

class DummyEnv(gym.Env):
    """A dummy class to workaround SB3's requirement for envs"""
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space


class ExtendedEnvWrapper(gym.Wrapper):
    """
    A wrapper that adds few additional important functions to environments
    for the Agents to use:

        reset(*args, **kwargs): allow reset function to recieve arbritrary arguments.
            (Note: Yes, this does not fix anything, but defined here to highlight that
            we want this for stuff like curriculum learning)
        get_current_observation: get the previous observation from step/reset.
        needs_reset: does environment need a reset (i.e. previous `done` from step).
        push_reward_accumulation: add a new tracker for tracking reward.
        pop_reward_accumulation: add a new tracker for tracking reward.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._reward_accumulation_stack = deque()
        self._last_obs = None
        self._needs_reset = True

    def get_current_observation(self) -> Optional[np.ndarray]:
        """Return the previous observation from step or reset, or None"""
        return self._last_obs

    def needs_reset(self) -> bool:
        """Returns True if environment needs resetting."""
        return self._needs_reset

    def reset(self, *args, **kwargs):
        self._needs_reset = False
        self._last_obs = self.env.reset(*args, **kwargs)
        return self._last_obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._last_obs = obs
        self._needs_reset = done
        for i in range(len(self._reward_accumulation_stack)):
            self._reward_accumulation_stack[i] += reward
        return obs, reward, done, info

    def push_reward_accumulation(self):
        """Push a new tracker for accumulating rewards"""
        self._reward_accumulation_stack.appendleft(0.0)

    def pop_reward_accumulation(self) -> float:
        """Pop the most recently pushed tracker for accumulated reward"""
        return self._reward_accumulation_stack.popleft()

    def reset_reward_accumulation(self):
        """Remove all trackers for reward accumulation"""
        self._reward_accumulation_stack.clear()


class LearningAgent():
    """
    A learning agent, which uses DQN under the hood to learn.

    Much like Gym environments, Agent has its own observation and action spaces
    which define (and help to tell to others) what kind of functions it uses to work.

    Subclasses should provde:
        - self.agent_observation_space (gym.spaces.Space)
        - self.agent_action_space (gym.spaces.Space)
    Along with these functions, subclasses have to define following functions:
        - self.env_observation_to_agent
        - self.env_action_to_agent
        - self.agent_action_to_env
        - self.reward
    By default these all correspond to environment's spaces (identical mapping).

    :param env_observation_space: environment's observation space
    :param env_action_space: environment's action space
    :param batch_size: batch size for a single DQN update (default 64).
    """
    def __init__(
        self,
        env_observation_space: gym.spaces.Space,
        env_action_space: gym.spaces.Space,
        batch_size: int = 64,
    ):
        # Default agent spaces to environment's
        self.agent_observation_space = env_observation_space
        self.agent_action_space = env_action_space
        self.env_observation_space = env_observation_space
        self.env_action_space = env_action_space

        self.rl_initialized = False
        self.dqn_agent = None
        self.batch_size = batch_size
        self.num_network_updates = 0

    def initialize_rl(self, use_cnn: bool = False, dqn_arguments: dict = None):
        """Initialize the RL part of this agent (i.e. create a DQN agent)

        NOTE: Important things to realize over SB3!
            - Epsilon is fixed at `exploration_initial_eps`
            - `target_update_rate` is in terms of updates, not stored experiences

        TODO add options to load agent parameters and such.

        :param dqn_arguments: arguments to the SB3 DQN agent. Only used if load_path=None. See `stable_baselines3.DQN` for more info.
        :param use_cnn: whether to use a CNN policy head for agent (default False). Only used if load_path=None.
        """
        assert not self.rl_initialized, "RL has already been initialized"
        self.rl_initialized = True
        self.num_network_updates = 0
        dummy_env = DummyEnv(self.agent_observation_space, self.agent_action_space)
        self.dqn_agent = DQN(
            "CnnPolicy" if use_cnn else "MlpPolicy",
            dummy_env,
            **dqn_arguments
        )
        self.dqn_agent.exploration_rate = self.dqn_agent.exploration_initial_eps

    def env_observation_to_agent(self, obs: np.ndarray) -> np.ndarray:
        """Transform observation from ``self.env_observation_space`` to ``self.agent_observation_space``

        :param obs: an observation from the environment.
        :return: observation transformed to conform with agent's observation space
        """
        assert self.agent_observation_space.contains(obs), "Environment observation is not a valid input for the agent. Override `env_observation_to_agent`"
        return obs

    def env_action_to_agent(self, action: np.ndarray) -> np.ndarray:
        """Transform observation from ``self.env_action_space`` to ``self.agent_action_space``

        Useful for imitation learning.

        :param action: an action from environment.
        :return: corresponding action in the agent's action space.
        """
        assert self.agent_action_space.contains(action), "Environment action is not a valid output of the agent. Override `env_action_to_agent`"
        return action

    def agent_action_to_env(self, action: np.ndarray) -> np.ndarray:
        """Transform observation from ``self.agent_action_space`` to ``self.env_action_space``

        :param action: an action from agent.
        :return: corresponding action in the environment's action space.
        """
        assert self.env_action_space.contains(action), "Agent action is not a valid input to the environment. Override `agent_action_to_env`"
        return action

    def store_experience(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        next_obs: np.ndarray
    ):
        assert self.rl_initialized, "You need to enable learning with `initialize_rl` method"
        # We need to add dummy dimensions because replay_buffer stores with "n_envs"
        self.dqn_agent.replay_buffer.add(
            obs[None],
            next_obs[None],
            action[None],
            reward,
            done,
            # Feed in empty dict
            infos=[{}]
        )

    def learn(self):
        """Do a learning update on the DQN agent, i.e. sample buffer of transitions and update the network."""
        assert self.rl_initialized, "You need to enable learning with `initialize_rl` method"
        # Make sure we have enough data in replay buffer before training
        if self.dqn_agent.replay_buffer.size() > self.dqn_agent.learning_starts:
            self.dqn_agent.train(1, batch_size=self.batch_size)
            self.num_network_updates += 1
            if self.num_network_updates % self.dqn_agent.target_update_interval == 0:
                polyak_update(
                    self.dqn_agent.q_net.parameters(),
                    self.dqn_agent.q_net_target.parameters(),
                    self.dqn_agent.tau
                )

    def get_action(self, agent_obs: np.ndarray, deterministic=True) -> np.ndarray:
        """Return an action for given agent observation.

        :param agent_obs:
        :param deterministic: should action be picked deterministically (always same for same input),
            or stochastically (e.g. a random action with small probability). Default True.
        :return: action corresponding to self.agent_action_space
        """
        assert self.rl_initialized, "You need to initialize RL with `initialize_rl` before using get_action method."
        return self.dqn_agent.predict(agent_obs, deterministic=deterministic)[0]

    def play(
        self,
        env: ExtendedEnvWrapper,
        termination_callback: Optional[Callable[[ExtendedEnvWrapper, ], bool]] = None,
        max_play_steps: Optional[int] = None,
        training: bool = True
    ):
        """The main function where this Agent takes control over the environment.

        This Agent plays in the environment until one of the conditions is met:
            - Agent itself is satisfied (returns)
            - ``max_play_steps`` have been played
            - ``termination_callback`` returns True
            - Episode ends (environment returns done=True)

        :param env:
        :param termination_callback: a callback function that takes in the environment, and if returns True,
            this function should return
        :param max_play_steps: maximum number of steps to play
        :param training: if samples should be stored for training and agent should be updated
        """
        assert self.rl_initialized, "You need to initialize RL with `initialize_rl` before using play method."
        agent_next_obs = None
        steps_played = 0
        while (
            not env.needs_reset() and
            (termination_callback is None or not termination_callback(env)) and
            (max_play_steps is None or steps_played < max_play_steps)
        ):
            steps_played += 1
            if agent_next_obs is None:
                env_obs = env.get_current_observation()
                agent_obs = self.env_observation_to_agent(env_obs)
            else:
                agent_obs = agent_next_obs

            # By default, you want to pick different actions during training for exploration
            agent_action = self.get_action(agent_obs, deterministic=not training)

            env_action = self.agent_action_to_env(agent_action)
            env_next_obs, reward, done, info = env.step(env_action)
            import time
            time.sleep(0.03)
            agent_next_obs = self.env_observation_to_agent(env_next_obs)

            # TODO need to inject reward shaping here somehow

            if training:
                self.store_experience(agent_obs, agent_action, reward, done, agent_next_obs)
                # For sample effeciency, we learn on each sample
                self.learn()
