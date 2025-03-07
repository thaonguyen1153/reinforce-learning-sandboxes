import gymnasium as gym
import blocksworld_env
import numpy as np
import os
import logging
import matplotlib
import matplotlib.pyplot as plt
import logging.handlers
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

#LOGGING CONFIG
filename = 'assn1_gymnasium_qlearning_original_hyperparameters.log'
logger = logging.getLogger()
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
logging.basicConfig(filename=filename, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG, filemode='w', force=True)

def plot_training_result(episode_rewards, episode_steps, gamma, epsilon, alpha, title):
    episodes = len(episode_rewards)  # Get the actual number of episodes from the data
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle( title, fontsize=16, fontweight='bold')

    episodes_range = range(1, episodes + 1)

    # Function to calculate moving average
    def moving_average(data, window):
        return np.convolve(data, np.ones(window), 'valid') / window

    # Rewards plot
    ax1.plot(episodes_range, episode_rewards, alpha=0.5)
    ax1.set_title('Rewards per Episode')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Steps plot
    ax2.plot(episodes_range, episode_steps, alpha=0.5)
    ax2.set_title('Steps per Episode')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Number of Steps')
    ax2.grid(True, linestyle='--', alpha=0.7)

    # Add annotations
    ax1.annotate(f'Max reward: {max(episode_rewards)} at episode {episode_rewards.index(max(episode_rewards)) + 1}',
                 xy=(0.05, 0.95), xycoords='axes fraction')
    ax2.annotate(f'Min steps: {min(episode_steps)} at episode {episode_steps.index(min(episode_steps)) + 1}',
                 xy=(0.05, 0.95), xycoords='axes fraction')

    # Add hyperparameters text box
    hyperparams_text = f"Hyperparameters:\n\nγ (gamma) = {gamma}\nε (epsilon) = {epsilon}\nα (alpha) = {alpha}"
    fig.text(0.87, 0.87, hyperparams_text, verticalalignment='top', horizontalalignment='left',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.subplots_adjust(top=0.88, right=0.85)  # Adjust right to make room for the text box
    plt.savefig('qlearning_blocksworld_original_hyperparameters.png')
    plt.show()
# create environment
env = gym.make('blocksworld_env/BlocksWorld-v0', render_mode="human")
observation, info = env.reset()

# QTable : contains the Q-Values for every (state,action) pair
# QTable Get the size of Qtable by getting the agent state
numstates = env.observation_space.n
numactions = env.action_space.n

qtable = np.random.rand(numstates, numactions).tolist()

logger.debug(f"Gymnasium established:\nQtable size: {numstates} x {numactions}")

#set the Qtable for the target state to 0 with all actions (state - action)
#qtable[47][0] = 0
#qtable[47][1] = 0
#qtable[47][2] = 0
#qtable[47][3] = 0

# hyperparameters --> this is important to fine tune the model--> this will change the behaviour of the agent
episodes = 50
gamma = 0.1 # discount values, if it approaches 1 it takes future rewards into account
epsilon = 0.08  # how much randomnes
decay = 0.1
alpha = 1 # step-size: how much of the new information is used, if it's 1 it's not save any last rewards

# Initialize variables before the training loop
shortest_steps = float('inf')  # Initialize with infinity
first_shortest_episode = None
# store rewards and steps for each episode
episode_rewards = []
episode_steps = []

# Here is the Agent
# training loop
for i in range(episodes):
    # state is aninteger
    state, info = env.reset()
    done = False
    logger.debug(f"state : {state} ")
    steps = 0
    accumulated_reward = 0  # add up reward

    while not done:
        os.system('clear')
        print("episode #", i+1, "/", episodes)
        # draw the grid
        env.render()

        # count steps to finish game
        steps += 1

        # act randomly sometimes to allow exploration
        # ε-greedy
        if np.random.uniform() < epsilon:
            action = env.action_space.sample()  # get a random action
            logger.debug(f"\nRandom action: {action}")
        # if not select max action in Qtable (act greedy)
        else:
            action = qtable[state].index(max(qtable[state]))
            logger.debug(f"\nMaxQ action: {action}")

        # take action
        next_state, reward, done, truncated, info = env.step(action)

        accumulated_reward += reward
        print(f"\nNext state: {next_state} current state: {state}, index using: action [{action}] steps with reward: {accumulated_reward}")
        logger.debug(f"\nNext state: {next_state} current state: {state}, index using: action [{action}] steps with reward: {accumulated_reward}")

        # update qtable value with Bellman equation
        # Q learning formula:
        # Q(S, A) <-  Q(S, A) + alpha * [ R +  gamma * (max Q(S', a)) -  Q(S, A)]
        qtable[state][action] = qtable[state][action] + alpha*(reward + gamma * max(qtable[next_state]) - qtable[state][action])
        # update state
        state = next_state

    episode_rewards.append(accumulated_reward)
    episode_steps.append(steps)

    # The more we learn, the less we take random actions
    epsilon -= decay*epsilon  # is not in the algorithm, it is the enhancement of the program

    # With this:
    if steps < shortest_steps:
        shortest_steps = steps
        first_shortest_episode = i + 1  # Adding 1 because episodes are typically counted from 1, not 0

    print(f"\nDone in {steps} steps with reward: {accumulated_reward}")
    logger.debug(f"\nDone in {steps} steps with reward: {accumulated_reward}")

# Print each result after an espisode is done
print(f"\nFinish training, shortest path found at espisode {first_shortest_episode} with {shortest_steps} steps")
logger.debug(f"\nFinish training, shortest path found at espisode {first_shortest_episode} with {shortest_steps} steps")
logger.info(f"Episode Rewards: {episode_rewards}")
logger.info(f"Episode Steps: {episode_steps}")
#After training, close the environment
env.close()
# Plot rewards and steps for each episode
plot_training_result(episode_rewards, episode_steps, gamma, epsilon, alpha, title="Original Hyperparameters" )
