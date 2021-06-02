#
# anssi-vizdoom.py
#
# Anssi's vision on how to combine dirty hacks and RL to train
# a bot that play singleplayer ViZDoom (and can be trained to do so)
#

# Expert data. Demonstrations in the varius maps
EXPERT_DATA = [...]


def navigation_agent_reward():
    if reached_goal:
        return 1
    elif collected_key:
        return 1
    elif agent_died:
        return -1
    else:
        return 0


def combat_agent_reward():
    if killed_enemy:
        return 1
    elif agent_died:
        return -1
    else:
        return 0


def main(is_training, starting_difficulty=0.0):
    navigation_agent = OffPolicyAgent(load_agent=not is_training)
    combat_agent = OffPolicyAgent(load_agent=not is_training)

    # Pretrain on all data.
    # E.g. fill replaybuffer and run bunch of training steps.
    # Also could keep a separate buffer for expert (e.g. like DQfD)
    if is_training:
        navigation_agent.pretrain(EXPERT_DATA, navigation_agent_reward)
        combat_agent.pretrain(EXPERT_DATA, combat_agent_reward)

    env = DoomGame()

    # Value in [0, 1], for a generic setting of difficulty.
    # 0.0 being super easy, 1.0 being regular settings
    difficulty = fixed_difficulty if fixed_difficulty is not None else 0.0

    while running:
        obs = env.reset(
            start_from_random_demonstration_point=EXPERT_DATA if is_training else None,
            randomize_enemy_healths=difficulty,
            randomize_player_health_speed_and_strength=difficulty,
            mirror=random_boolean() if is_training
        )

        done = False
        while not done:
            agent = combat_agent if enemys_in_sight() else navigation_agent

            # Keep track where agent has been, and keep track of some kind of
            # heatmap
            update_obs_with_visited_locations(obs)

            action = agent.predict(obs)

            if agent is combat_agent:
                # If difficulty is low enough, help agent to shoot
                if random() > difficulty and enemy_in_sights():
                    action.fire_weapon = True
                # If even lower, help aiming
                if difficulty < 0.2 and not enemy_in_sights():
                    change_action_mouse_movement(action, closest_target())

            # Reward is done by the functions
            new_obs, _, done, info = env.step(action)

            if is_training:
                if agent is combat_agent:
                    reward = combat_agent_reward()
                else:
                    reward = navigation_agent_reward()
                agent.store_experience(obs, action, new_obs, reward, done)

                # Adjust difficulty
                if game_over and completed_game:
                    difficulty += 0.01
                elif game_over and (agent_died or timeout):
                    difficulty -= 0.01

            obs = new_obs

    save_trained_agents(navigation_agent, combat_agent)


if __name__ == '__main__':
    # First train
    main(is_training=True)
    # And then enjoy
    main(is_training=False, starting_difficulty=1.0)
