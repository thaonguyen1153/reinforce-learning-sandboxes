from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import random
from screen import Display
from swiplserver import PrologMQI,PrologThread

class BlocksWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        # Run Prolog interpreter and load blocks world
        self.mqi = PrologMQI()
        self.prolog_thread = self.mqi.create_thread()
        result = self.prolog_thread.query('[blocks_world]')
        if not result:
            raise RuntimeError("Could not load blocks_world.pl.")

        # Calling query to return all of the possible states
        prolog_states = self.prolog_thread.query('state(State)')
        # construct the state dictionary
        # will create a dict like:
        # {’bc2’:0, ’bc3’:1, …,…}
        self.states_dict = {state['State']: index for index, state in enumerate(prolog_states)}

        # Set up a dictionary to convert action numbers into prolog actions
        self.actions_dict = {}
        # Call query action(A) to get all actions
        result = self.prolog_thread.query("action(A)")
        # result is like this, where the first action is move(a,b,c)
        # [{'A': {'args': ['a', 'b', 'c'], 'functor': 'move'}},...]
        for i, A in enumerate(result):
            # get the action name (as a functor)
            if isinstance(A['A'], str):  # case where {'A': '_'}
                break
            else:
                action_string = A['A']['functor']
                # indicate the first argument
                first = True
                for arg in A['A']['args']:
                    if first:
                        # already have a first, next will not be first
                        first = False
                        action_string += '('
                    else:
                        action_string += ','

                    # concat all arguments into functor predicate
                    action_string += str(arg)
                action_string += ')'
                self.actions_dict[i] = action_string

        # Observation space is the length of state dict
        self.observation_space = spaces.Discrete(len(self.states_dict))

        # there is only one action: move
        self.action_space = spaces.Discrete(len(self.actions_dict))

        # initial starting state of the blocks
        self.state = 0 # the fist state
        # initial_state will return like 'bc2'
        self.initial_state_str = self.get_state_str(self.state)

        # choose random target which is not the start state
        self.target_state_str = self.get_random_target_state() # will set self.target_state to a random state string
        self.target_state = self.states_dict[self.target_state_str]
        # render mode

        # Initialize PyGame display if render_mode is "human"
        self.render_mode = render_mode
        if self.render_mode == "human":
            self.display = Display()

        self.window = None
        self.clock = None

    def get_random_target_state(self):
        # Choose a random state that is not the initial state
        possible_targets = [state for state in self.states_dict.keys() if state != self.initial_state_str]
        return random.choice(possible_targets)

    def get_state_str(self, state_int):
        return list(self.states_dict.keys())[list(self.states_dict.values()).index(state_int)]

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Reset to the initial state
        # a. Randomly set a new target state
        self.target_state_str = self.get_random_target_state()
        self.target_state = self.states_dict[self.target_state_str]

        # Set the target in the display if it exists
        if self.render_mode == "human":
            self.display.target = self.target_state_str

        # b. Issue Prolog query to reset, get back to the initial state
        self.prolog_thread.query("reset")

        # c. Retrieve the current state from Prolog
        result = list(self.prolog_thread.query("current_state(State)"))
        if result:
            current_state_string = result[0]['State']
            self.state = self.states_dict[current_state_string]
        else:
            raise RuntimeError("Failed to retrieve current state from Prolog")

        # Prepare observation
        observation = self.state

        # Prepare info dict
        info = {}

        return observation, info

    def step(self, action):
        # Convert action integer to action string
        action_string = self.actions_dict[action]
        # a. Issue Prolog query to step/1 predicate
        # for example: step(move(c,a,3)): move c from a to 3
        step_result = self.prolog_thread.query(f"step({action_string})")

        # b. Check the result of step/1 predicate
        if step_result:
            # Action was possible, retrieve new state
            state_result = list(self.prolog_thread.query("current_state(State)"))
            if state_result:
                # move and update state
                current_state_string = state_result[0]['State']
                self.state = self.states_dict[current_state_string]
                if self.render_mode == "human":
                    self.display.step(current_state_string)
            else:
                raise RuntimeError("Failed to retrieve current state from Prolog")
            reward = -1
        else:
            # Action was not possible
            reward = -10

        # Check if the target state is reached
        done = (self.state == self.target_state)

        # c. If done, set reward to 100
        if done:
            reward = 100

        # Prepare observation and info
        observation = self.state
        info = {}

        return observation, reward, done, False, info

    def render(self):
        if self.render_mode is None:
            return

        if self.render_mode == "human":
            if not hasattr(self, 'display'):
                raise RuntimeError("Display not initialized. Make sure to set render_mode='human' in the constructor.")

            # Convert current state integer to state string
            current_state_string = self.get_state_str(self.state)

            # Update the display
            self.display.step(current_state_string)

    def close(self):
        # shutdown prolog server
        if self.mqi:
            self.mqi.stop()

        # close pygame
        if hasattr(self, 'display'):
            self.display.close()
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
