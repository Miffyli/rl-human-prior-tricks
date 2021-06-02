from argparse import ArgumentParser
import time
from collections import OrderedDict

import numpy as np
import gym
import torch as th
from torch import nn
import vizdoom as vz

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import get_linear_fn

from envs.vizdoom_environment import DoomEnvironment, AppendFeaturesToImageWrapper, TransposeWrapper, DeathmatchRewardWrapper
from agent import LearningAgent, ExtendedEnvWrapper

parser = ArgumentParser("Run ViZDoom experiments")
parser.add_argument(
    "--agent-type",
    type=str,
    choices=["vanilla-dqn", "reward-shaping", "manual-actions", "manual-actions-reward-shaping", "manual-hierarchy"],
    help="Algorithm to train",
    required=True
)
args = parser.parse_args()

th.set_num_threads(1)
CONFIG = "doom_scenarios/deathmatch.cfg"

MAX_TRAINING_STEPS = int(5e6)
TRAIN_EPISODES = 10
EVAL_EPISODES = 10
FRAMESKIP = 4

# Hacky linear schedule for exploration
# Same as in stable-baselines3
EXPLORATION_INITIAL_EPS = 1.0
EXPLORATION_FINAL_EPS = 0.05
EXPLORATION_RATIO = 0.1

DQN_ARGUMENTS = dict(
    learning_starts=10000,
    target_update_interval=10000,
    exploration_initial_eps=0.1,
    device="cuda",
)

# Reward for increasing a game variable.
# No negatives, because might lead to avoid
# shooting.
# This also has the potential of reward loops...
VIZDOOM_INCREASE_REWARD = OrderedDict((
    (vz.GameVariable.HITCOUNT, 0.1),
    (vz.GameVariable.HEALTH, 0.1),
    (vz.GameVariable.ARMOR, 0.1),
    (vz.GameVariable.AMMO0, 0.1),
    (vz.GameVariable.AMMO1, 0.1),
    (vz.GameVariable.AMMO2, 0.1),
    (vz.GameVariable.AMMO3, 0.1),
    (vz.GameVariable.AMMO4, 0.1),
    (vz.GameVariable.AMMO5, 0.1),
    (vz.GameVariable.AMMO6, 0.1),
    (vz.GameVariable.AMMO7, 0.1),
    (vz.GameVariable.AMMO8, 0.1),
    (vz.GameVariable.AMMO9, 0.1)
))


class StepCounterWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.step_counter = 0

    def step(self, action):
        self.step_counter += 1
        return self.env.step(action)


class RewardShapingLearningAgent(LearningAgent):
    def reward_shaping_before_step(self, env):
        self.pre_step_variables = dict((k, env.doomgame.get_game_variable(k)) for k in VIZDOOM_INCREASE_REWARD)

    def compute_reward_shaping(self, env):
        rs = 0
        for k, reward in VIZDOOM_INCREASE_REWARD.items():
            diff = env.doomgame.get_game_variable(k) - self.pre_step_variables[k]
            if diff > 0:
                rs += reward
        return rs

    def play(self, env, termination_callback=None, training=True, max_play_steps=None):
        assert self.rl_initialized, "You need to initialize RL with `initialize_rl` before using play method."
        agent_next_obs = None
        steps_played = 0
        while (
            not env.needs_reset() and
            (termination_callback is None or not termination_callback(env)) and
            (max_play_steps is None or steps_played < max_play_steps)
        ):
            steps_played += 1
            # TODO we could reuse "agent_next_obs" from previous loop here
            if agent_next_obs is None:
                env_obs = env.get_current_observation()
                agent_obs = self.env_observation_to_agent(env_obs)
            else:
                agent_obs = agent_next_obs

            # By default, you want to pick different actions during training for exploration
            agent_action = self.get_action(agent_obs, deterministic=not training)

            self.reward_shaping_before_step(env)
            env_action = self.agent_action_to_env(agent_action)
            env_next_obs, reward, done, info = env.step(env_action)
            agent_next_obs = self.env_observation_to_agent(env_next_obs)

            reward += self.compute_reward_shaping(env)
            if training:
                self.store_experience(agent_obs, agent_action, reward, done, agent_next_obs)
                # For sample effeciency, we learn on each sample
                self.learn()


class ManualActionLearningAgent(LearningAgent):
    """
    Agent that fires the weapon if
    enemy hitbox is under crosshair
    (only training)
    """
    def play(self, env, termination_callback=None, training=True, max_play_steps=None):
        assert self.rl_initialized, "You need to initialize RL with `initialize_rl` before using play method."
        agent_next_obs = None
        steps_played = 0
        while (
            not env.needs_reset() and
            (termination_callback is None or not termination_callback(env)) and
            (max_play_steps is None or steps_played < max_play_steps)
        ):
            steps_played += 1
            # TODO we could reuse "agent_next_obs" from previous loop here
            if agent_next_obs is None:
                env_obs = env.get_current_observation()
                agent_obs = self.env_observation_to_agent(env_obs)
            else:
                agent_obs = agent_next_obs

            # By default, you want to pick different actions during training for exploration
            agent_action = self.get_action(agent_obs, deterministic=not training)

            if training:
                # Do not do manual actions while evaluating, only during training
                # Check if enemy is undercrosshair
                if env.is_deathmatch_enemy_under_crosshair():
                    agent_action = env.get_action_for_attack()

            env_action = self.agent_action_to_env(agent_action)
            env_next_obs, reward, done, info = env.step(env_action)
            agent_next_obs = self.env_observation_to_agent(env_next_obs)

            if training:
                self.store_experience(agent_obs, agent_action, reward, done, agent_next_obs)
                # For sample effeciency, we learn on each sample
                self.learn()


class ManualActionRewardShapingLearningAgent(LearningAgent):
    """
    Manual actions + reward shaping
    """
    def reward_shaping_before_step(self, env):
        self.pre_step_variables = dict((k, env.doomgame.get_game_variable(k)) for k in VIZDOOM_INCREASE_REWARD)

    def compute_reward_shaping(self, env):
        rs = 0
        for k, reward in VIZDOOM_INCREASE_REWARD.items():
            diff = env.doomgame.get_game_variable(k) - self.pre_step_variables[k]
            if diff > 0:
                rs += reward
        return rs

    def play(self, env, termination_callback=None, training=True, max_play_steps=None):
        assert self.rl_initialized, "You need to initialize RL with `initialize_rl` before using play method."
        agent_next_obs = None
        steps_played = 0
        while (
            not env.needs_reset() and
            (termination_callback is None or not termination_callback(env)) and
            (max_play_steps is None or steps_played < max_play_steps)
        ):
            steps_played += 1
            # TODO we could reuse "agent_next_obs" from previous loop here
            if agent_next_obs is None:
                env_obs = env.get_current_observation()
                agent_obs = self.env_observation_to_agent(env_obs)
            else:
                agent_obs = agent_next_obs

            # By default, you want to pick different actions during training for exploration
            agent_action = self.get_action(agent_obs, deterministic=not training)

            if training:
                # Do not do manual actions while evaluating, only during training
                # Check if enemy is undercrosshair
                if env.is_deathmatch_enemy_under_crosshair():
                    agent_action = env.get_action_for_attack()
                    # Dirtyest of the hardcodings: If we have action mapping
                    # thing, do mapping here
                    if hasattr(self, "inverse_action_mapping"):
                        agent_action = np.array(self.inverse_action_mapping[agent_action.item()])

            self.reward_shaping_before_step(env)
            env_action = self.agent_action_to_env(agent_action)
            env_next_obs, reward, done, info = env.step(env_action)
            agent_next_obs = self.env_observation_to_agent(env_next_obs)

            reward += self.compute_reward_shaping(env)

            if training:
                self.store_experience(agent_obs, agent_action, reward, done, agent_next_obs)
                # For sample effeciency, we learn on each sample
                self.learn()


class ManualHierarchyLearningAgent(LearningAgent):
    """
    Agent that uses manual hierarchy and two agents: one for shooting (turning and aiming), second for movement
    """
    def __init__(
        self,
        shooting_agent,
        navigation_agent,
        env_observation_space: gym.spaces.Space,
        env_action_space: gym.spaces.Space,
        batch_size: int = 64,
        steps_per_subagent: int = 1,
    ):
        super().__init__(env_observation_space, env_action_space, batch_size)

        self.steps_per_subagent = 1
        self.shooting_agent = shooting_agent
        self.navigation_agent = navigation_agent

    def play(self, env, termination_callback=None, training=True, max_play_steps=None):
        steps_played = 0
        while (
            not env.needs_reset() and
            (termination_callback is None or not termination_callback(env)) and
            (max_play_steps is None or steps_played < max_play_steps)
        ):
            steps_played += 1
            if env.is_deathmatch_enemy_visible():
                self.shooting_agent.play(env, training=training, max_play_steps=self.steps_per_subagent)
            else:
                self.navigation_agent.play(env, training=training, max_play_steps=self.steps_per_subagent)


def create_augmented_nature_cnn_class(num_features):
    class AugmentedNatureCNN(BaseFeaturesExtractor):
        def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
            super(AugmentedNatureCNN, self).__init__(observation_space, features_dim)
            n_input_channels = observation_space.shape[0] - 1
            self.cnn = nn.Sequential(
                nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.Flatten(),
            )

            # Compute shape by doing one forward pass
            with th.no_grad():
                n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None, :-1]).float()).shape[1]

            self.linear = nn.Sequential(nn.Linear(n_flatten + num_features, features_dim), nn.ReLU())

        def forward(self, observations: th.Tensor) -> th.Tensor:
            return self.linear(
                th.cat(
                    (
                        # Keep dim
                        self.cnn(observations[:, :-1]),
                        observations[:, 1, 0, :num_features]
                    ),
                    dim=1
                )
            )

    return AugmentedNatureCNN


def main_vanilla_dqn():
    env = DoomEnvironment(
        CONFIG,
        allowed_buttons="deathmatch",
        discrete_max_buttons_down=2,
        only_screen_buffer=False,
        frame_skip=4
    )
    env = StepCounterWrapper(env)
    env = DeathmatchRewardWrapper(env, reward_multiplier=1.0)
    env = AppendFeaturesToImageWrapper(env)
    env = TransposeWrapper(env)
    env = ExtendedEnvWrapper(env)

    my_cnn_class = create_augmented_nature_cnn_class(env.num_features)

    DQN_ARGUMENTS["policy_kwargs"] = dict(
        features_extractor_class=my_cnn_class
    )

    agent = LearningAgent(
        env.observation_space,
        env.action_space
    )
    agent.initialize_rl(dqn_arguments=DQN_ARGUMENTS, use_cnn=True)

    exploration_schedule = get_linear_fn(EXPLORATION_INITIAL_EPS, EXPLORATION_FINAL_EPS, EXPLORATION_RATIO)

    train_steps = 0
    train_episodes = 0
    start_time = time.time()
    while train_steps < MAX_TRAINING_STEPS:
        # Update exploration ratio
        agent.dqn_agent.exploration_rate = exploration_schedule(1 - (train_steps / MAX_TRAINING_STEPS))
        before_train_steps = env.step_counter
        for _ in range(TRAIN_EPISODES):
            _ = env.reset()
            agent.play(env)
            train_episodes += 1
        train_steps += (env.step_counter - before_train_steps) * FRAMESKIP

        eval_rewards = 0
        for _ in range(EVAL_EPISODES):
            _ = env.reset()
            env.push_reward_accumulation()
            agent.play(env, training=False)
            eval_rewards += env.pop_reward_accumulation()
        print("time-seconds {:<12}; train-episodes {:<10}; train-steps {:<12}; average-reward {:<7.2f}".format(
            int(time.time() - start_time),
            train_episodes,
            train_steps,
            eval_rewards / EVAL_EPISODES
        ))


def main_reward_shaping():
    env = DoomEnvironment(
        CONFIG,
        allowed_buttons="deathmatch",
        discrete_max_buttons_down=2,
        only_screen_buffer=False,
        frame_skip=4
    )
    env = StepCounterWrapper(env)
    env = DeathmatchRewardWrapper(env, reward_multiplier=1.0)
    env = AppendFeaturesToImageWrapper(env)
    env = TransposeWrapper(env)
    env = ExtendedEnvWrapper(env)

    my_cnn_class = create_augmented_nature_cnn_class(env.num_features)

    DQN_ARGUMENTS["policy_kwargs"] = dict(
        features_extractor_class=my_cnn_class
    )

    agent = RewardShapingLearningAgent(
        env.observation_space,
        env.action_space
    )
    agent.initialize_rl(dqn_arguments=DQN_ARGUMENTS, use_cnn=True)

    exploration_schedule = get_linear_fn(EXPLORATION_INITIAL_EPS, EXPLORATION_FINAL_EPS, EXPLORATION_RATIO)

    train_steps = 0
    train_episodes = 0
    start_time = time.time()
    while train_steps < MAX_TRAINING_STEPS:
        # Update exploration ratio
        agent.dqn_agent.exploration_rate = exploration_schedule(1 - (train_steps / MAX_TRAINING_STEPS))
        before_train_steps = env.step_counter
        for _ in range(TRAIN_EPISODES):
            _ = env.reset()
            agent.play(env)
            train_episodes += 1
        train_steps += (env.step_counter - before_train_steps) * FRAMESKIP

        eval_rewards = 0
        for _ in range(EVAL_EPISODES):
            _ = env.reset()
            env.push_reward_accumulation()
            agent.play(env, training=False)
            eval_rewards += env.pop_reward_accumulation()
        print("time-seconds {:<12}; train-episodes {:<10}; train-steps {:<12}; average-reward {:<7.2f}".format(
            int(time.time() - start_time),
            train_episodes,
            train_steps,
            eval_rewards / EVAL_EPISODES
        ))


def main_manual_actions():
    env = DoomEnvironment(
        CONFIG,
        allowed_buttons="deathmatch",
        discrete_max_buttons_down=2,
        only_screen_buffer=False,
        frame_skip=4
    )
    env = StepCounterWrapper(env)
    env = DeathmatchRewardWrapper(env, reward_multiplier=1.0)
    env = AppendFeaturesToImageWrapper(env)
    env = TransposeWrapper(env)
    env = ExtendedEnvWrapper(env)

    my_cnn_class = create_augmented_nature_cnn_class(env.num_features)

    DQN_ARGUMENTS["policy_kwargs"] = dict(
        features_extractor_class=my_cnn_class
    )

    agent = ManualActionLearningAgent(
        env.observation_space,
        env.action_space
    )
    agent.initialize_rl(dqn_arguments=DQN_ARGUMENTS, use_cnn=True)

    exploration_schedule = get_linear_fn(EXPLORATION_INITIAL_EPS, EXPLORATION_FINAL_EPS, EXPLORATION_RATIO)

    train_steps = 0
    train_episodes = 0
    start_time = time.time()
    while train_steps < MAX_TRAINING_STEPS:
        # Update exploration ratio
        agent.dqn_agent.exploration_rate = exploration_schedule(1 - (train_steps / MAX_TRAINING_STEPS))
        before_train_steps = env.step_counter
        for _ in range(TRAIN_EPISODES):
            _ = env.reset()
            agent.play(env)
            train_episodes += 1
        train_steps += (env.step_counter - before_train_steps) * FRAMESKIP

        eval_rewards = 0
        for _ in range(EVAL_EPISODES):
            _ = env.reset()
            env.push_reward_accumulation()
            agent.play(env, training=False)
            eval_rewards += env.pop_reward_accumulation()
        print("time-seconds {:<12}; train-episodes {:<10}; train-steps {:<12}; average-reward {:<7.2f}".format(
            int(time.time() - start_time),
            train_episodes,
            train_steps,
            eval_rewards / EVAL_EPISODES
        ))


def main_manual_actions_reward_shaping():
    env = DoomEnvironment(
        CONFIG,
        allowed_buttons="deathmatch",
        discrete_max_buttons_down=2,
        only_screen_buffer=False,
        frame_skip=4
    )
    env = StepCounterWrapper(env)
    env = DeathmatchRewardWrapper(env, reward_multiplier=1.0)
    env = AppendFeaturesToImageWrapper(env)
    env = TransposeWrapper(env)
    env = ExtendedEnvWrapper(env)

    my_cnn_class = create_augmented_nature_cnn_class(env.num_features)

    DQN_ARGUMENTS["policy_kwargs"] = dict(
        features_extractor_class=my_cnn_class
    )

    agent = ManualActionRewardShapingLearningAgent(
        env.observation_space,
        env.action_space
    )
    agent.initialize_rl(dqn_arguments=DQN_ARGUMENTS, use_cnn=True)

    exploration_schedule = get_linear_fn(EXPLORATION_INITIAL_EPS, EXPLORATION_FINAL_EPS, EXPLORATION_RATIO)

    train_steps = 0
    train_episodes = 0
    start_time = time.time()
    while train_steps < MAX_TRAINING_STEPS:
        # Update exploration ratio
        agent.dqn_agent.exploration_rate = exploration_schedule(1 - (train_steps / MAX_TRAINING_STEPS))
        before_train_steps = env.step_counter
        for _ in range(TRAIN_EPISODES):
            _ = env.reset()
            agent.play(env)
            train_episodes += 1
        train_steps += (env.step_counter - before_train_steps) * FRAMESKIP

        eval_rewards = 0
        for _ in range(EVAL_EPISODES):
            _ = env.reset()
            env.push_reward_accumulation()
            agent.play(env, training=False)
            eval_rewards += env.pop_reward_accumulation()
        print("time-seconds {:<12}; train-episodes {:<10}; train-steps {:<12}; average-reward {:<7.2f}".format(
            int(time.time() - start_time),
            train_episodes,
            train_steps,
            eval_rewards / EVAL_EPISODES
        ))


def main_manual_hierarchy():
    env = DoomEnvironment(
        CONFIG,
        allowed_buttons="deathmatch",
        discrete_max_buttons_down=2,
        only_screen_buffer=False,
        frame_skip=4
    )
    env = StepCounterWrapper(env)
    env = DeathmatchRewardWrapper(env, reward_multiplier=1.0)
    env = AppendFeaturesToImageWrapper(env)
    env = TransposeWrapper(env)
    env = ExtendedEnvWrapper(env)

    my_cnn_class = create_augmented_nature_cnn_class(env.num_features)

    DQN_ARGUMENTS["policy_kwargs"] = dict(
        features_extractor_class=my_cnn_class
    )

    # Two agents: shooting and navigation.

    # Use manual actions with the hierarchy
    shooting_agent = ManualActionRewardShapingLearningAgent(
        env.observation_space,
        env.action_space,
    )
    navigation_agent = RewardShapingLearningAgent(
        env.observation_space,
        env.action_space
    )

    shooting_agent.initialize_rl(dqn_arguments=DQN_ARGUMENTS, use_cnn=True)
    navigation_agent.initialize_rl(dqn_arguments=DQN_ARGUMENTS, use_cnn=True)

    agent = ManualHierarchyLearningAgent(shooting_agent, navigation_agent, env.observation_space, env.action_space)

    exploration_schedule = get_linear_fn(EXPLORATION_INITIAL_EPS, EXPLORATION_FINAL_EPS, EXPLORATION_RATIO)

    train_steps = 0
    train_episodes = 0
    start_time = time.time()
    while train_steps < MAX_TRAINING_STEPS:
        # Update exploration ratio
        shooting_agent.dqn_agent.exploration_rate = exploration_schedule(1 - (train_steps / MAX_TRAINING_STEPS))
        navigation_agent.dqn_agent.exploration_rate = exploration_schedule(1 - (train_steps / MAX_TRAINING_STEPS))
        before_train_steps = env.step_counter
        for _ in range(TRAIN_EPISODES):
            _ = env.reset()
            agent.play(env)
            train_episodes += 1
        train_steps += (env.step_counter - before_train_steps) * FRAMESKIP

        eval_rewards = 0
        for _ in range(EVAL_EPISODES):
            _ = env.reset()
            env.push_reward_accumulation()
            agent.play(env, training=False)
            eval_rewards += env.pop_reward_accumulation()
        print("time-seconds {:<12}; train-episodes {:<10}; train-steps {:<12}; average-reward {:<7.2f}".format(
            int(time.time() - start_time),
            train_episodes,
            train_steps,
            eval_rewards / EVAL_EPISODES
        ))


if __name__ == "__main__":
    if args.agent_type == "vanilla-dqn":
        main_vanilla_dqn()
    elif args.agent_type == "reward-shaping":
        main_reward_shaping()
    elif args.agent_type == "manual-actions":
        main_manual_actions()
    elif args.agent_type == "manual-actions-reward-shaping":
        main_manual_actions_reward_shaping()
    elif args.agent_type == "manual-hierarchy":
        main_manual_hierarchy()
