# A test for LearningAgent to see that it works
import gym
import \
    torch as th

from agent import LearningAgent, ExtendedEnvWrapper

th.set_num_threads(1)
ENV = "CartPole-v1"
EPISODES = 10000

DQN_ARGUMENTS = dict(
    learning_starts=1000,
    target_update_interval=1000,
    exploration_initial_eps=0.1,
    device="cpu",
)


def main():
    env = gym.make(ENV)
    env = gym.wrappers.TransformReward(env, lambda r: r * 0.01)
    env = ExtendedEnvWrapper(env)

    agent = LearningAgent(
        env.observation_space,
        env.action_space
    )
    agent.initialize_rl(dqn_arguments=DQN_ARGUMENTS)

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


if __name__ == "__main__":
    main()
