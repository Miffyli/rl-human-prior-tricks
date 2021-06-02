#  vizdoom_environment.py
#  ViZDoom wrapped into Environment object (similar to OpenAI Gym API)
#

import numpy as np
import vizdoom as vz
import gym
from gym import spaces
import itertools
import cv2

# label object_names of the enemies
DEATHMATCH_ENEMY_NAMES = set([
    "ShotgunGuy",
    "ChaingunGuy",
    "Zombieman",
    "MarineChainsawVzd",
    "HellKnight",
    "Demon"
])

# Mapping of [-1, 1] continuous
# action to how many degrees agent
# turns per tick
MOUSE_SPEED = 45

# Map GameVariables to functions that take in
# said GameVariable and return something more
# convenient for networks (one-hots, in range [0,1],
# etc)
GAME_VARIABLE_PROCESSOR = {
    vz.GameVariable.HEALTH: lambda health: min(health, 200) / 100,
    vz.GameVariable.ARMOR: lambda armor: min(armor, 200) / 200,
    vz.GameVariable.SELECTED_WEAPON_AMMO: lambda ammo: int(ammo > 0),
    vz.GameVariable.SELECTED_WEAPON: lambda weapon: weapon,
    vz.GameVariable.AMMO0: lambda ammo: int(ammo > 0),
    vz.GameVariable.AMMO1: lambda ammo: int(ammo > 0),
    vz.GameVariable.AMMO2: lambda ammo: int(ammo > 0),
    vz.GameVariable.AMMO3: lambda ammo: int(ammo > 0),
    vz.GameVariable.AMMO4: lambda ammo: int(ammo > 0),
    vz.GameVariable.AMMO5: lambda ammo: int(ammo > 0),
    vz.GameVariable.AMMO6: lambda ammo: int(ammo > 0),
    vz.GameVariable.AMMO7: lambda ammo: int(ammo > 0),
    vz.GameVariable.AMMO8: lambda ammo: int(ammo > 0),
    vz.GameVariable.AMMO9: lambda ammo: int(ammo > 0),
    vz.GameVariable.KILLCOUNT: lambda count: min(count, 100) / 20,
}

# Different sets of allowed buttons
BUTTON_SETS = {
    "bare-minimum": (
        vz.Button.MOVE_FORWARD, vz.Button.TURN_LEFT
    ),
    "minimal": (
        vz.Button.MOVE_FORWARD, vz.Button.TURN_LEFT, vz.Button.TURN_RIGHT
    ),
    "backward": (
        vz.Button.MOVE_FORWARD, vz.Button.MOVE_BACKWARD,
        vz.Button.TURN_LEFT, vz.Button.TURN_RIGHT
    ),
    "strafe": (
        vz.Button.MOVE_FORWARD, vz.Button.MOVE_BACKWARD,
        vz.Button.TURN_LEFT, vz.Button.TURN_RIGHT,
        vz.Button.MOVE_LEFT, vz.Button.MOVE_RIGHT
    ),
    # For environments that require shooting
    "bare-minimum-attack": (
        vz.Button.MOVE_FORWARD, vz.Button.TURN_LEFT, vz.Button.ATTACK
    ),
    "minimal-attack": (
        vz.Button.MOVE_FORWARD, vz.Button.TURN_LEFT, vz.Button.TURN_RIGHT,
        vz.Button.ATTACK
    ),
    "backward-attack": (
        vz.Button.MOVE_FORWARD, vz.Button.MOVE_BACKWARD,
        vz.Button.TURN_LEFT, vz.Button.TURN_RIGHT,
        vz.Button.ATTACK
    ),
    "strafe-attack": (
        vz.Button.MOVE_FORWARD, vz.Button.MOVE_BACKWARD,
        vz.Button.TURN_LEFT, vz.Button.TURN_RIGHT,
        vz.Button.MOVE_LEFT, vz.Button.MOVE_RIGHT,
        vz.Button.ATTACK
    ),
    # Go bananas
    "all": (
        vz.Button.MOVE_FORWARD, vz.Button.MOVE_BACKWARD,
        vz.Button.TURN_LEFT, vz.Button.TURN_RIGHT,
        vz.Button.MOVE_LEFT, vz.Button.MOVE_RIGHT,
        vz.Button.LOOK_UP, vz.Button.LOOK_DOWN,
        vz.Button.MOVE_UP, vz.Button.MOVE_DOWN,
        vz.Button.ATTACK, vz.Button.USE,
        vz.Button.JUMP, vz.Button.CROUCH,
        vz.Button.TURN180, vz.Button.ALTATTACK,
        vz.Button.RELOAD, vz.Button.ZOOM,
        vz.Button.SPEED, vz.Button.STRAFE,
        vz.Button.LAND, vz.Button.SELECT_WEAPON0,
        vz.Button.SELECT_WEAPON1, vz.Button.SELECT_WEAPON2,
        vz.Button.SELECT_WEAPON3, vz.Button.SELECT_WEAPON4,
        vz.Button.SELECT_WEAPON5, vz.Button.SELECT_WEAPON6,
        vz.Button.SELECT_WEAPON7, vz.Button.SELECT_WEAPON8,
        vz.Button.SELECT_WEAPON9, vz.Button.SELECT_WEAPON0,
        vz.Button.SELECT_NEXT_WEAPON, vz.Button.SELECT_PREV_WEAPON,
        vz.Button.DROP_SELECTED_WEAPON, vz.Button.ACTIVATE_SELECTED_ITEM,
        vz.Button.SELECT_NEXT_ITEM, vz.Button.SELECT_PREV_ITEM,
        vz.Button.DROP_SELECTED_WEAPON
    ),
    # Deathmatch specific
    "deathmatch": (
        vz.Button.MOVE_FORWARD, vz.Button.MOVE_BACKWARD,
        vz.Button.TURN_LEFT, vz.Button.TURN_RIGHT,
        vz.Button.MOVE_LEFT, vz.Button.MOVE_RIGHT,
        vz.Button.TURN180, vz.Button.ATTACK,
        vz.Button.SELECT_NEXT_WEAPON, vz.Button.SELECT_PREV_WEAPON,
        vz.Button.SPEED,
    ),
    # Defend-the-center
    "defend-the-center": (
        vz.Button.TURN_LEFT, vz.Button.TURN_RIGHT, vz.Button.ATTACK
    ),
}


def create_discrete_actions(num_buttons, max_buttons_down):
    """
    Return list of available actions, when we have
    num_buttons buttons available and we are allowed
    to press at most max_buttons_down, as
    a discrete action space
    """
    # Remove no-op action
    actions = [
        list(action) for action in itertools.product((0, 1), repeat=num_buttons)
        if (sum(action) <= max_buttons_down and sum(action) > 0)
    ]
    return actions


class DeathmatchRewardWrapper(gym.Wrapper):
    """
    A wrapper that fixes Deathmatch reward by checking
    if player really made a hit when they got a kill.

    Modifies the original reward by only giving it when
    player really made a kill, and then multiplies it with
    the given weight.
    """
    def __init__(self, env, reward_multiplier=0.1):
        super().__init__(env)
        self.reward_multiplier = reward_multiplier
        self.last_killcount = 0
        self.last_hitcount = 0

    def reset(self):
        self.last_killcount = 0
        self.last_hitcount = 0
        return self.env.reset()

    def step(self, action):
        s, r, d, i = self.env.step(action)
        killcount = self.env.doomgame.get_game_variable(vz.GameVariable.KILLCOUNT)
        hitcount = self.env.doomgame.get_game_variable(vz.GameVariable.HITCOUNT)
        # Make sure it is us who does the kill
        # TODO does this work with launchers and other stuff...?
        if hitcount > self.last_hitcount:
            r = r * self.reward_multiplier
        else:
            # Player did not make the kill
            r = 0
        self.last_killcount = killcount
        self.last_hitcount = hitcount
        return s, r, d, i


class TransposeWrapper(gym.Wrapper):
    """
    Transpose channel location
    """
    def __init__(self, env):
        super().__init__(env)

        shape = self.env.observation_space.shape
        self.observation_space = spaces.Box(
            low=self.env.observation_space.low.transpose(2, 0, 1),
            high=self.env.observation_space.high.transpose(2, 0, 1),
            shape=(shape[2], shape[0], shape[1]),
            dtype=self.env.observation_space.dtype
        )

    def reset(self):
        return self.env.reset().transpose(2, 0, 1)

    def step(self, action):
        s, r, d, i = self.env.step(action)
        return s.transpose(2, 0, 1), r, d, i


class AppendFeaturesToImageWrapper(gym.Wrapper):
    """
    Append direct features to the image observation on last channel

    Assumes the underlying observation space is a Tuple of (image_obs, feature_obs)
    """

    def __init__(self, env):
        super().__init__(env)

        # Check that observation_space is valid
        if not isinstance(env.observation_space, spaces.Tuple) or len(env.observation_space) != 2:
            raise ValueError("Underlying observation_space should be a tuple of two spaces")

        self.env = env
        self.image_height = env.observation_space[0].shape[0]
        self.image_width = env.observation_space[0].shape[1]
        self.original_channels = env.observation_space[0].shape[2]
        self.num_image_values = self.image_height * self.image_width
        self.num_features = env.observation_space[1].shape[0]

        self.num_padding = self.num_image_values - self.num_features

        # Make sure image is large enough to store the direct features
        if self.num_padding < 0:
            raise ValueError("Images are too small to contain all features ({})".format(
                self.num_features
            ))

        self.data_dtype = env.observation_space[0].dtype

        self.observation_space = spaces.Box(
            # Hardcoded values!
            low=0,
            high=255 if self.data_dtype == np.uint8 else 0,
            shape=(self.image_height, self.image_width, self.original_channels + 1),
            dtype=self.data_dtype
        )

    def _append_features_to_image(self, image, features):
        """
        Append append features on a new channel in the image
        """
        # Turn features to same size as number of values in image channel,
        # resize and append to image

        # Unnormalize data
        # A trick for stable-baselines3
        if self.data_dtype == np.uint8:
            features = np.clip(features * 255, 0, 255).astype(np.uint8)

        features = np.concatenate((
            features,
            np.zeros((self.num_padding,), dtype=self.data_dtype)
        ))

        features.resize((self.image_height, self.image_width, 1))

        image = np.concatenate((image, features), axis=2)

        return image

    def step(self, action):
        obs, reward, terminal, info = self.env.step(action)
        obs = self._append_features_to_image(obs[0], obs[1])
        return obs, reward, terminal, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        obs = self._append_features_to_image(obs[0], obs[1])
        return obs


class DoomEnvironment(gym.Env):
    def __init__(
        self,
        config,
        allowed_buttons="minimal",
        action_space_type="discrete",
        discrete_max_buttons_down=1,
        frame_skip=4,
        image_height=60,
        image_width=80,
        only_screen_buffer=True,
        normalize_screen_buffer=False
    ):
        """
        Create the ViZDoom game and define settings (do not init yet)
        Parameters:
            config: String specifying path to the config file
            allowed_buttons: One of "bare-minimum", "minimal", "backward" or
                "strafe", indicating the set of allowed buttons.
                See BUTTON_SETS.
            action_space_type: What type of action-space we are using, can be one of
                "discrete", "multidiscrete" or "mouse", where "mouse" is a multi-discrete
                action-space, with "TURN_LEFT/RIGHT" replaced with a continuous
                mouse movement.
            frame_skip: Amount of frames one action will repeated for
            max_buttons_down: How many buttons we can press down at once
            image_height/width: Image shape after resizing
            only_screen_buffer: Only return screen buffer (image) observation
            normalize_screen_buffer: Return image pixels in [0, 1], otherwise in [0, 255]
        """
        super().__init__()

        # Ad-hoc safeguard to avoid running bad experiments
        if "health_gathering" in config and only_screen_buffer:
            raise ValueError("Health gathering tasks do not work with only screen buffer")
        if action_space_type == "mouse" and "mouse" not in allowed_buttons:
            raise ValueError("Select mouse-### button-set for 'mouse' action space")

        self.config = config
        self.frame_skip = frame_skip
        self.image_width = image_width
        self.image_height = image_height
        self.only_screen_buffer = only_screen_buffer
        self.normalize_screen_buffer = normalize_screen_buffer

        # Create game
        self.doomgame = vz.DoomGame()
        self.doomgame.load_config(self.config)
        self.init_done = False

        self.image_channels = self.doomgame.get_screen_channels()
        # Simple boolean to check if we even need to call
        # resize functions
        self.need_resize = True
        # Premade tuple for cv2.resize
        self.target_size = (self.image_width, self.image_height)
        self.doom_image_width = self.doomgame.get_screen_width()
        self.doom_image_height = self.doomgame.get_screen_height()
        # TODO not correct if weapon/HUD is visible! It is slightly offset
        self.doom_crosshair_location_x = self.doom_image_width / 2
        self.doom_crosshair_location_y = self.doom_image_height / 2
        if (self.image_width == self.doom_image_width and
                self.image_height == self.doom_image_height):
            self.need_resize = False

        self.allowed_buttons = BUTTON_SETS[allowed_buttons]
        self.num_buttons = len(self.allowed_buttons)
        for button in self.allowed_buttons:
            self.doomgame.add_available_button(button)

        if action_space_type == "mouse":
            self.doomgame.add_available_button(vz.Button.TURN_LEFT_RIGHT_DELTA)

        # Create actions according to action_space_type
        self.action_space_type = action_space_type
        self.action_handler = None
        if self.action_space_type == "discrete":
            # Discrete actions. Flatten buttons with maximum of
            # "discrete_max_buttons_down" buttons down at the same
            # time
            self.action_list = create_discrete_actions(
                self.num_buttons,
                discrete_max_buttons_down
            )
            self.num_actions = len(self.action_list)
            self.action_space = spaces.Discrete(self.num_actions)
            self.action_handler = self._build_discrete_action
        elif self.action_space_type == "multidiscrete":
            # Multidiscrete actions, one 0/1 decision per button
            self.action_space = spaces.MultiDiscrete([2] * self.num_buttons)
            self.action_handler = self._build_multidiscrete_action
        elif self.action_space_type == "mouse":
            # Same as multidiscrete, but we have continuous action
            # for mouse movement.
            self.action_space = spaces.Tuple(
                spaces.MultiDiscrete([2] * self.num_buttons),
                spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
            )
            self.action_handler = self._build_multidiscrete_mouse_action
        else:
            raise ValueError("Unknown action space '%s'" % self.action_space_type)

        # Handle GameVariables
        self.game_variables = self.doomgame.get_available_game_variables()
        self.game_variable_processors = [
            GAME_VARIABLE_PROCESSOR[game_variable] for game_variable in self.game_variables
        ]
        self.num_game_variables = len(self.game_variables)

        image_max_value = 1 if normalize_screen_buffer else 255
        image_dtype = np.float32 if normalize_screen_buffer else np.uint8
        if only_screen_buffer:
            self.observation_space = spaces.Box(
                0, image_max_value, shape=(self.image_height, self.image_width, self.image_channels), dtype=image_dtype,
            )
        else:
            self.observation_space = spaces.Tuple((
                spaces.Box(0, image_max_value, shape=(self.image_height, self.image_width, self.image_channels), dtype=image_dtype),
                spaces.Box(0, 1, shape=(self.num_game_variables,), dtype=np.float32)
            ))

        self.last_observation = None
        self.last_state = None

        # A dumb ad-hoc cache for these experiments
        self._attack_action = None

        print(self.num_actions)

    def _get_mapping_from_allowed_buttons(self, allowed_buttons):
        """
        Return a dictionary that maps a smaller, discrete action-space actions
        to the original action space of this action. Only allowed buttons given as
        a list are used in this action space
        """
        output_dict = {}
        allowed_button_indeces = [self.allowed_buttons.index(button) for button in allowed_buttons]
        for original_action_index in range(self.num_actions):
            # Check that only allowed buttons are allowed in the action
            action_list = self.action_list[original_action_index]
            allowed_button_sum = sum(action_list[allowed_button_index] for allowed_button_index in allowed_button_indeces)
            if sum(action_list) == allowed_button_sum:
                output_dict[len(output_dict)] = original_action_index
        return output_dict

    def get_shooting_agent_action_mapping(self):
        """
        Return dictionary that maps discrete actions of a smaller space
        to actions that are used for shooting/combat in this full action space

        Attacking agent is allowed to turn, shoot, change weapons and strafe
        """
        allowed_buttons = [
            vz.Button.ATTACK,
            vz.Button.MOVE_LEFT,
            vz.Button.MOVE_RIGHT,
            vz.Button.TURN_LEFT,
            vz.Button.TURN_RIGHT,
            vz.Button.SELECT_NEXT_WEAPON,
            vz.Button.SELECT_PREV_WEAPON,
        ]
        return self._get_mapping_from_allowed_buttons(allowed_buttons)

    def get_navigation_agent_action_mapping(self):
        """
        Return dictionary that maps discrete actions of a smaller space
        to actions that are used for moving/navigating

        Attacking agent is allowed to move forward/backward, speed, turn180
        """
        allowed_buttons = [
            vz.Button.MOVE_FORWARD,
            vz.Button.MOVE_BACKWARD,
            vz.Button.SPEED,
            vz.Button.TURN180,
            vz.Button.TURN_LEFT,
            vz.Button.TURN_RIGHT,
        ]
        return self._get_mapping_from_allowed_buttons(allowed_buttons)

    def is_deathmatch_enemy_visible(self):
        """
        Returns True if an enemy character is visible.

        Only works for Deathmatch.wad scenario
        """
        if self.last_state is None:
            return False
        for doom_object in self.last_state.labels:
            if doom_object.object_name in DEATHMATCH_ENEMY_NAMES:
                return True
        return False

    def is_deathmatch_enemy_under_crosshair(self):
        """
        Return True if an enemy is under crosshair currently.

        Only works for Deathmatch.wad scenario
        """
        if self.last_state is None:
            return False
        for doom_object in self.last_state.labels:
            if doom_object.object_name in DEATHMATCH_ENEMY_NAMES:
                x = doom_object.x
                y = doom_object.y
                x2 = x + doom_object.width
                y2 = y + doom_object.height
                if x <= self.doom_crosshair_location_x <= x2:
                    if y <= self.doom_crosshair_location_y <= y2:
                        return True
        return False

    def get_action_for_attack(self):
        """
        Return the action that corresponds to current action space
        and only contains "ATTACK=1".

        Only for Discrete action space
        """
        if self._attack_action is not None:
            return self._attack_action
        assert self.action_space_type == "discrete"
        attack_index = self.allowed_buttons.index(vz.Button.ATTACK)
        for i, action in enumerate(self.action_list):
            if sum(action) == 1 and action[attack_index] == 1:
                attack_action = np.array(i)
                self._attack_action = attack_action
                return attack_action

    def _build_discrete_action(self, action):
        """
        Return ViZDoom action for discrete action space
        """
        return self.action_list[action]

    def _build_multidiscrete_action(self, action):
        """
        Return ViZDoom action for multidiscrete action space
        """
        # Conveniently enough, vizdoom accepts its actions
        # in a multi-discrete form, soooo...
        return action.tolist()

    def _build_multidiscrete_mouse_action(self, action):
        """
        Return ViZDoom action for multidiscrete + continous
        action space
        """
        # TURN_LEFT_RIGHT_DELTA is last action for ViZDoom.

        vizdoom_actions = self._build_multidiscrete_action[action[0]]
        mouse_turn = action[1].item()

        # Map [-1, 1] to [-MOUSE_SPEED, MOUSE_SPEED] turn.
        # NOTE that this is per-tick amount of turning, i.e.
        #      final amount of turning is multiplied by frame_skip
        mouse_turn = int(mouse_turn * MOUSE_SPEED)

        vizdoom_actions += [mouse_turn]

        return vizdoom_actions

    def _preprocess_state(self, state):
        """
        Preprocess vizdoom state (one from DoomGame.get_state)

        Return observation according to self.observation_space
        """
        image = state.screen_buffer

        # Add channel dim if using Gray
        if image.ndim == 2:
            image = image[None]

        # Transpose to format almost everything else uses
        # (C, H, W) -> (H, W, C)
        image = np.transpose(image, (1, 2, 0))

        # Resize if necessary
        if self.need_resize:
            image = cv2.resize(image, self.target_size)
            # Add image channel again if cv2 removed it...
            if image.ndim == 2:
                image = image[..., None]

        # And normalize
        if self.normalize_screen_buffer:
            image = image.astype(np.float32) / 255.0

        # If we do not want to have game variables (we only need screen buffer),
        # return here already before fiddling with game variables
        if self.only_screen_buffer:
            return image

        # Get game variables and process them
        game_vars = state.game_variables
        processed_vars = np.zeros((self.num_game_variables,), dtype=np.float32)
        for i in range(len(game_vars)):
            processed_vars[i] = self.game_variable_processors[i](game_vars[i])

        return (image, processed_vars)

    def initialize(self):
        """
        Initialize the game
        """
        self.init_done = True
        self.doomgame.init()

    def step(self, action):
        action = self.action_handler(action)
        reward = self.doomgame.make_action(action, self.frame_skip)
        terminal = self.doomgame.is_episode_finished()
        observation = None
        if terminal:
            # No observation available,
            # give the previous observation
            observation = self.last_observation
        else:
            self.last_state = self.doomgame.get_state()
            observation = self._preprocess_state(self.last_state)
        # Keep track of the last_observation
        # in case we hit end of the episode
        # (no state available, give last_observation instead)
        self.last_observation = observation
        return observation, reward, terminal, {}

    def reset(self):
        if not self.init_done:
            self.initialize()
        self.doomgame.new_episode()
        self.last_state = self.doomgame.get_state()
        observation = self._preprocess_state(self.last_state)
        self.last_observation = observation
        return observation

    def close(self):
        self.doomgame.close()

    def seed(self, seed):
        self.doomgame.set_seed(seed)
