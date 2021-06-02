import gym
import torch as th

from discrete_sac import SAC

def main():
    env = gym.wrappers.TransformReward(gym.make("LunarLander-v2"), lambda r: r)
    model = SAC(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=[128, 128]),
        learning_starts=10000,
        verbose=2,
        buffer_size=1000000,

    )
    model.learn(total_timesteps=int(5e5))


if __name__ == "__main__":
    main()
