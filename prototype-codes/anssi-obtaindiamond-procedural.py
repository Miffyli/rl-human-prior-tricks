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

# This version is more of a library thing (coder calls things of the library)
# Pros:
#   - Coder has exact control over how environment is treated
# Cons:
#   - Need to write the training loop (steps and all that)
#   - Risk of mistakes (e.g. not calling learn with right data, not updating "obs", etc)

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

            # Alternatively we play until desired item is obtained
            action = agent.step(obs)

            new_obs, reward, done, info = env.step(action)

            if is_training:
                agent.store_experience(obs, action, new_obs, reward, done)

if __name__ == '__main__':
    # First train
    main(is_training=True)
    # And then enjoy
    main(is_training=False)
