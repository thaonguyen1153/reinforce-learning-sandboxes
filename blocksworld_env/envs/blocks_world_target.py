from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import random
from screen import Display
from swiplserver import PrologMQI,PrologThread

class BlocksWorldTargetEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        # Run Prolog interpreter and load blocks world
        self.mqi = PrologMQI()
        self.prolog_thread = self.mqi.create_thread()
        result = self.prolog_thread.query('[blocks_world_target]')
        if not result:
            raise RuntimeError("Could not load blocks_world_target.pl.")

        # Calling query to return all of the possible states
        prolog_states = self.prolog_thread.query('state(State)')

        # state dict will create a dict like:
        # {’bc1bc1’:0, ’bc1bc2’:1, …,…}
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
        # initial_state will return like 'bc1bc2', so the initial state is the first set of 2 characters
        self.initial_state_str, _ = self.split_state(self.get_state_str(self.state))

        # Initial, target is the same as initial state
        self.target_state_str = self.get_random_target_state()  # will set self.target_state to a random state string
        self.target_state = self.states_dict[self.target_state_str]

        # Initialize PyGame display if render_mode is "human"
        self.render_mode = render_mode
        if self.render_mode == "human":
            self.display = Display()

        self.window = None
        self.clock = None

    def get_random_target_state(self):
        # Get all possible target states (last 3 characters of each state)
        all_targets = [state[-3:] for state in self.states_dict.keys()]

        # Remove the initial state from possible targets
        possible_targets = [target for target in all_targets if target != self.initial_state_str[-3:] and len(target) > 2]

        # Choose a random target state
        random_target = random.choice(possible_targets)

        # Combine the current state (first 3 characters of initial state) with the new target
        return self.initial_state_str[:3] + random_target

    def get_state_str(self, state_int):
        return list(self.states_dict.keys())[list(self.states_dict.values()).index(state_int)]

    def split_state(self, state_string):
        if len(state_string) != 6:
            raise ValueError("State string must be exactly 6 characters long")

        current_state = state_string[:3]
        target_state = state_string[3:]

        return current_state, target_state

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Reset to the initial state
        # a. Randomly set a new target state
        self.target_state_str = self.get_random_target_state() # string 6 characters
        self.target_state = self.states_dict[self.target_state_str] # integer
        _, target_state_3c = self.split_state(self.target_state_str)

        # Set the target in the display if it exists
        if self.render_mode == "human":
            self.display.target = target_state_3c

        # b. Issue Prolog query to reset
        self.prolog_thread.query("reset")

        # c. Retrieve the current state from Prolog
        # the current state remains 3 characters
        result = list(self.prolog_thread.query("current_state(State)"))
        if result:
            current_state_string = str(result[0]['State'])
            _, target_state_3c = self.split_state(self.target_state_str)
            self.state = self.states_dict[current_state_string + target_state_3c]  # self.state is the index
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
                _, target_state_3c = self.split_state(self.target_state_str)
                self.state = self.states_dict[current_state_string + target_state_3c] #self.state is the index
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
        observation =  self.state

        info = {}

        return observation, reward, done, False, info

    def render(self):
        if self.render_mode is None:
            return

        if self.render_mode == "human":
            if not hasattr(self, 'display'):
                raise RuntimeError("Display not initialized. Make sure to set render_mode='human' in the constructor.")

            # Convert current state integer to state string
            current_state_str_3c, _ = self.split_state(self.get_state_str(self.state))

            # Update the display
            self.display.step(current_state_str_3c)

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
