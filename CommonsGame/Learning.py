import numpy as np
import gym

from constants import tinyMap, smallMap, bigMap, SUSTAINABILITY_MATTERS

"""
TODO: Decide the configuration parameters of the environment. We give some example ones.
"""

number_of_agents = 1

def new_action_space(action_space):
    """
        TODO:  Modify this method if you want to limit the action space of the agents. This method
        is specially important if you are using Tabular RL

    :param action_space: a list of integers
    :return: a new list of integers, smaller or equal
    """

    # Example: To remove the possibility of shooting, do
    action_space.remove(env.SHOOT)
    action_space.remove(env.TURN_CLOCKWISE)
    action_space.remove(env.TURN_COUNTERCLOCKWISE)

    return action_space


def new_state(agent, state, tabularRL):
    """
    TODO: Modify the state (if you are using Tabular RL) to simplify it so it can be useful to the agent
    Ideally you will be able to create a map from every state to a different integer number
    :param agent: an integer to know which agent it is
    :param state: a list of integers
    :param tabularRL: boolean to know if you are using tabularRL or deep RL
    :return: a new list of integers, smaller or equal
    """

    if tabularRL:
        # Provisionally you have a very basic encoding, very hard-coded for tinyMap:

        if len(state) == 0:
            return 8*3*8
        else:
            # Obtain the agent's position:
            agent_x = state[-1-4*number_of_agents+4*agent]
            agent_y = state[-4*number_of_agents+4*agent]
            position = agent_x + 2*agent_y  # we encode them as a scalar, there are 8 different positions

            # Obtain the agent's amount of apples, but we only consider if it has 0, 1 or more
            agent_apples = min(state[1-4*number_of_agents+4*agent], 2)  # 3 different values

            # Apple states, we know which ones they are in tinyMap, so we look for each of them if there is an apple:
            apple_state_1 = int(state[1] == 64)
            apple_state_2 = int(state[2] == 64)
            apple_state_3 = int(state[5] == 64)
            #apple_state_4 = int(state[6] == 64)

            where_apples = apple_state_1 + 2*(apple_state_2 + 2*(apple_state_3)) # +2*apple_state_4)) # we encode them as a scalar, 16 different values

            # Total number of states: 8x3x16 = 384 states (or 385 if we count the state of being ill)
            position_and_apples = position + 8*(agent_apples + 3*where_apples)

            return position_and_apples
    else:
        return state


def learning_policy(state, action_space, Q_function, epsilon):
    """
    Epsilon-greedy policy.

    :param state:  typically, a list of integers
    :param action_space: a list of integers representing the possible actions that an agent can do
    :param Q_function: a numpy 2-d array (in tabular RL) that contains information about each state-action pair
    :return: a recommended action
    """

    choose_randomly = np.random.random() < epsilon

    if choose_randomly:
        index = np.random.randint(low=0, high=len(action_space))
    else:
        index = np.argmax(Q_function[state][action_space])

    return action_space[index]


def update_Q(Q, s, a, r, s_prima, alpha, gamma):
    """

    As a template, it updates Q(s,a) with the current reward

    :param Q_function: a numpy 2-d array (in tabular RL) that contains information about each state-action pair
    :param current_state: a list of integers
    :param current_action: an integer
    :param current_reward: an integer
    :param next_state: a list of integers
    :return: an updated Q_function
    """

    Q[s][a] = alpha * (r + gamma * np.max(Q[s_prima]) - Q[s][a])

    return Q


def learning_loop(environment, tabularRL, learning_rate=0.2, discount_factor=0.8):
    """
    Adapted for Q-learning

    :param environment: the environment already configured
    :param tabularRL: boolean to know if you will be using tabular RL or deep RL
    :return:
    """

    n_agent_cells = 3*4
    n_apples = 3
    apples_in_ground = 3**2

    len_state_space = n_agent_cells*n_apples*apples_in_ground + 1  # You need to change this!!! This is provisional and only works for tinyMap

    total_action_space = [i for i in range(environment.action_space.n)]
    action_space = new_action_space(total_action_space)

    Q_functions = np.zeros((number_of_agents, len_state_space, environment.action_space.n))

    episodes = 100
    timesteps = 1
    epsilon = 0.9

    for episode in range(episodes):
        initial_state = environment.reset(common_pool=1, apples_yes_or_not=[False, True, True])
        print(initial_state)
        state = []

        for agent in range(number_of_agents):
            state.append(new_state(agent, initial_state[agent], tabularRL))
        print(state)
        for timestep in range(timesteps):

            print("--Episode", episode ,", Time step", timestep, "--")
            environment.render()
            nActions = list()

            for agent in range(number_of_agents):
                action_i = learning_policy(state[agent], action_space, Q_functions[agent], epsilon)
                nActions.append(action_i)

            nObservations, nRewards, nDone, nInfo = environment.step(nActions)

            print("Reward : ", nRewards[0])


            next_state = []

            for agent in range(number_of_agents):
                next_state.append(new_state(agent, nObservations[agent], tabularRL))
                Q_functions[agent] = update_Q(Q_functions[agent], state[agent], nActions[agent], nRewards[agent], next_state[agent], learning_rate, discount_factor)

            state = next_state

            if all(nDone):
                break

    return Q_functions


if __name__ == '__main__':
    tabularRL = True
    env = gym.make('CommonsGame:CommonsGame-v0', numAgents=number_of_agents, mapSketch=tinyMap, visualRadius=3, fullState=False, tabularState=tabularRL, agent_pos=[2, 2])
    learning_loop(env, tabularRL)

