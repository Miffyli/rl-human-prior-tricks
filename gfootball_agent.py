# A test for LearningAgent to see that it works
import os
from datetime import datetime

import gfootball.env as football_env
import gym
import numpy as np
import torch as th
from stable_baselines3 import DQN
from stable_baselines3.common.utils import polyak_update

from agent import LearningAgent, ExtendedEnvWrapper, DummyEnv

th.set_num_threads(1)
ENV = "11_vs_11_easy_stochastic"
EPISODES = 3333
CHANGE_ENV_AFTER_EPISODES = 10

CURRICULUM = True
HIERARCHICAL = False

DQN_ARGUMENTS = dict(
    learning_starts=100,
    target_update_interval=2500,
    exploration_initial_eps=0.01,
    device="cpu",
    learning_rate=0.0000115,
    buffer_size=10000,
    batch_size=512,
    gamma=0.999,
    optimize_memory_usage=True,
)


def create_single_football_env():
    """Creates gfootball environment."""
    env = football_env.create_environment(
        env_name=ENV, stacked=False,
        rewards='scoring,checkpoint',
        representation='simple115v2',
        logdir='./logs/gfootball',
        write_goal_dumps=False,
        write_full_episode_dumps=False,
        render=False,
        dump_frequency=0,
        other_config_options={'action_set': 'v2'}
    )

    return env


def create_curriculum_football_env(difficulty):
    """Creates gfootball environment."""

    assert ENV in ['11_vs_11_easy_stochastic', '11_vs_11_stochastic', '11_vs_11_hard_stochastic']
    assert difficulty >= 0
    assert difficulty <= 1

    env = football_env.create_environment(
        env_name=ENV, stacked=False,
        rewards='scoring,checkpoint',
        representation='simple115v2',
        logdir='./logs/gfootball',
        write_goal_dumps=False,
        write_full_episode_dumps=False,
        render=False,
        dump_frequency=0,
        other_config_options={'action_set': 'v2', 'right_team_difficulty': difficulty}
    )

    return env


class ImageToPytorch(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageToPytorch, self).__init__(env)
        old_shape = self.env.observation_space.shape
        new_shape = (old_shape[-1], old_shape[0], old_shape[1])
        self.observation_space = gym.spaces.Box(low=self.env.observation_space.low[0][0][0],
                                                high=self.env.observation_space.high[0][0][0], shape=new_shape,
                                                dtype=self.env.observation_space.dtype)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


class GfootballLearningAgent(LearningAgent):
    def __init__(self, env_observation_space: gym.spaces.Space, env_action_space: gym.spaces.Space):
        super().__init__(env_observation_space, env_action_space)
        self.agent_to_env_action_mapping = {i: i for i in range(20)}
        self.env_to_agent_action_mapping = {v: k for k, v in self.agent_to_env_action_mapping.items()}
        self.agent_action_space = gym.spaces.Discrete(len(self.agent_to_env_action_mapping))

        self.dqn_attack_agent = None
        self.dqn_defend_agent = None

    def env_action_to_agent(self, action: np.ndarray) -> np.ndarray:
        """Transform observation from ``self.env_action_space`` to ``self.agent_action_space``

        Useful for imitation learning.

        :param action: an action from environment.
        :return: corresponding action in the agent's action space.
        """
        assert self.agent_action_space.contains(
            action), "Environment action is not a valid output of the agent. Override `env_action_to_agent`"
        return self.env_to_agent_action_mapping[action]

    def agent_action_to_env(self, action: np.int64) -> np.int64:
        """Transform observation from ``self.agent_action_space`` to ``self.env_action_space``

        :param action: an action from agent.
        :return: corresponding action in the environment's action space.
        """
        assert self.env_action_space.contains(
            action), "Agent action is not a valid input to the environment. Override `agent_action_to_env`"
        return self.agent_to_env_action_mapping[action]

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
        self.dqn_attack_agent = DQN(
            "CnnPolicy" if use_cnn else "MlpPolicy",
            dummy_env,
            **dqn_arguments
        )
        if HIERARCHICAL:
            self.dqn_defend_agent = DQN(
                "CnnPolicy" if use_cnn else "MlpPolicy",
                dummy_env,
                **dqn_arguments
            )

    def store_experience(
            self,
            obs: np.ndarray,
            action: np.ndarray,
            reward: float,
            done: bool,
            next_obs: np.ndarray
    ):
        assert self.rl_initialized, "You need to enable learning with `initialize_rl` method"
        if self.__is_attacking(obs) or not HIERARCHICAL:
            # We need to add dummy dimensions because replay_buffer stores with "n_envs"
            self.dqn_attack_agent.replay_buffer.add(
                obs[None],
                next_obs[None],
                action[None],
                reward,
                done
            )
        else:
            self.dqn_defend_agent.replay_buffer.add(
                obs[None],
                next_obs[None],
                action[None],
                reward,
                done
            )

    def learn(self):
        """Do a learning update on the DQN agent, i.e. sample buffer of transitions and update the network."""
        assert self.rl_initialized, "You need to enable learning with `initialize_rl` method"
        agents = [self.dqn_attack_agent, self.dqn_defend_agent] if HIERARCHICAL else [self.dqn_attack_agent]
        for agent in agents:
            # Make sure we have enough data in replay buffer before training
            if agent.replay_buffer.size() > agent.learning_starts:
                agent.train(1, batch_size=self.batch_size)
                self.num_network_updates += 1
                if self.num_network_updates % (len(agents) * agent.target_update_interval) == 0:
                    polyak_update(
                        agent.q_net.parameters(),
                        agent.q_net_target.parameters(),
                        agent.tau
                    )

    def get_action(self, agent_obs: np.ndarray, deterministic=True) -> np.ndarray:
        """Return an action for given agent observation.

        :param agent_obs:
        :param deterministic: should action be picked deterministically (always same for same input),
            or stochastically (e.g. a random action with small probability). Default True.
        :return: action corresponding to self.agent_action_space
        """
        assert self.rl_initialized, "You need to initialize RL with `initialize_rl` before using get_action method."
        if self.__is_attacking(agent_obs) or not HIERARCHICAL:
            return self.dqn_attack_agent.predict(agent_obs, deterministic=deterministic)[0]
        else:
            return self.dqn_defend_agent.predict(agent_obs, deterministic=deterministic)[0]

    @staticmethod
    def __is_attacking(obs):
        return obs[96] == 1


class Logger:
    def __init__(self, dir: str):
        now = datetime.now().strftime('%Y-%m-%d_%H_%M')
        self.dir = os.path.join(dir, now)
        os.makedirs(self.dir, exist_ok=True)
        self.file = open(os.path.join(self.dir, 'log.txt'), mode='w', encoding='utf-8')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()

    def log(self, entry: str):
        self.file.write(entry)

    def __del__(self):
        self.file.close()


def main():
    difficulty = 0.0
    if CURRICULUM:
        env = create_curriculum_football_env(difficulty)
    else:
        env = create_single_football_env()
    env = ExtendedEnvWrapper(env)

    agent = GfootballLearningAgent(
        env.observation_space,
        env.action_space
    )
    agent.initialize_rl(use_cnn=False, dqn_arguments=DQN_ARGUMENTS)

    with Logger(f'./experiments/full_game_CL{CURRICULUM}_HL{HIERARCHICAL}') as logger:
        for episode_i in range(EPISODES):
            _ = env.reset()
            # Play one train episode
            agent.play(env)
            # Play one eval episode
            _ = env.reset()
            env.push_reward_accumulation()
            agent.play(env, training=False)
            episode_reward = env.pop_reward_accumulation()
            print("Episode {}, reward {}".format(episode_i, episode_reward))
            logger.log(f"{episode_i};{episode_reward};{difficulty}\n")

            if episode_i % CHANGE_ENV_AFTER_EPISODES == 0 and CURRICULUM:
                # Increase difficulty
                difficulty += .05 / (EPISODES / CHANGE_ENV_AFTER_EPISODES)
                env = ExtendedEnvWrapper(create_curriculum_football_env(difficulty))


if __name__ == "__main__":
    main()
