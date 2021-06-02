#
# anssi-obtaindiamond.py
#
# Anssi's vision on how to combine dirty hacks and RL to train
# a bot that completes the ObtainDiamond task.
#

# Idea is from the Align-RUDDER paper:
# - One agent per item type
# - Manual item hierarchy that tells what we should get next
# - Initialize game from different locations (in item hierarchy)
#   to ensure we train all agents.

# This is version using callbacks (the library is a framework)
# Pros:
#   - Can not mess up the training loop
#   - Setup of callbacks could be taken from stable-baselines3 (e.g. on_step)
# Cons:
#   - Hmm could use some new functions in env as we keep passing it around
#   - Might be hard to see how all agents connect to each other (hidden by class structure)
#     (see e.g. Costa's arguments for "one file which contains everything")
#   


class Env(gym.Env):
    """An extensions to the gym.Env"""

    def get_current_observation(self):
        """Return the current state of the game"""


class Agent():
    """One individual agent that learns and controls things.

    For hierarchy, this Agent can pass the "play" to any subsquent agents.
    """

    def process_observation(self, obs):
        """Transform an observation from environment into something this agent uses"""
        return obs

    def env_action_to_agent(self, action):
        """Modify action from the environment (e.g. dataset) into something this agent uses.

        Useful for imitation learning
        """
        raise NotImplementedError()

    def agent_action_to_env(self, action):
        """Modify action from the agent into something what environment uses"""
        return action

    def play(self, env, termination_callback, training):
        """Play and learn until callback breaks or environment returns done (?)"""
        while not env.needs_reset() and termination_callback(env):
            obs = env.get_current_observation()
            agent_obs = self.process_observation(obs)
            action = self.get_action(agent_obs)
            # This part could be replaced with calling an another
            # agent
            next_obs, reward, done, info = env.step(action)
            # TODO need to inject reward shaping here somehow
            if training:
                self.store_experience(obs, next_obs, reward, done)


def main(is_training):
    # Initialize one agent per item type
    obtain_x_agents = [initialize_or_load_agent(load_agent=not is_training)]
    item_hierarchy = ItemHierarchy()

    env = MineRLEnv()

    while running:
        obs = env.reset(
            start_from_random_position=pick_random_start_location() if is_training else pick_new_start()
        )

        done = False
        while not done:
            # Pick agent we need for the next item
            item_we_need_next = item_hierarchy.next_required_item(item_hierarchy)
            agent = obtain_x_agents[item_we_need_next]

            agent.play(
                env,
                training=is_training,
                termination_callback=callback_for_item_reached(item_we_need_next)
            )

if __name__ == '__main__':
    # First train
    main(is_training=True)
    # And then enjoy
    main(is_training=False)
