import numpy as np
import gym

from constants import tinyMap, smallMap, bigMap, TOO_MANY_APPLES
from new_utils import *


def learning_policy(state, action_space, Q_function, epsilon, mode, weights):
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
        if mode == "lex":
            _, _, index = lexicographic_Qs(action_space, Q_function[state][action_space])

            return index
        elif mode == "scalarisation":
            index = np.argmax(scalarised_Qs(len(action_space), Q_function[state][action_space], weights))
        else:
            print("Uh oh!!!!!")
            index = 0

    return action_space[index]


def update_Q(Q, s, a, r, s_prima, alpha, gamma, action_space, mode, weights):
    """

    As a template, it updates Q(s,a) with the current reward

    :param Q_function: a numpy 2-d array (in tabular RL) that contains information about each state-action pair
    :param current_state: a list of integers
    :param current_action: an integer
    :param current_reward: an integer
    :param next_state: a list of integers
    :return: an updated Q_function
    """

    if mode == "lex":
        V_ind, V_eth, _ = lexicographic_Qs(action_space, Q[s_prima][action_space])
        V_prima = np.array([V_ind, V_eth])
    elif mode == "scalarisation":
        index = np.argmax(scalarised_Qs(len(action_space), Q[s_prima][action_space], weights))
        V_prima = Q[s_prima][action_space[index]]
    else:
        print("Uh oh!!!")
        V_prima = [0., 0.]

    Q[s][a] = alpha * (r + gamma * V_prima - Q[s][a])

    return Q


def learning_loop(environment, tabularRL, learning_rate=0.2, discount_factor=0.8, mode="scalarisation", weights=[1.0, 0.0]):
    """
    Adapted for Q-learning

    :param environment: the environment already configured
    :param tabularRL: boolean to know if you will be using tabular RL or deep RL
    :return:
    """

    Q_functions = np.zeros((number_of_agents, len_state_space, environment.action_space.n, number_of_objectives))
    total_action_space = [i for i in range(environment.action_space.n)]
    action_space = new_action_space(total_action_space, environment)


    episodes = 200
    timesteps = 5000
    epsilon = 0.6

    for episode in range(episodes):

        print("--Episode", episode)
        initial_state = environment.reset()
        state = []

        for agent in range(number_of_agents):
            state.append(new_state(agent, initial_state[agent], tabularRL))

        print(Q_functions[0][state[0]], Q_functions[1][state[1]])
        for timestep in range(timesteps):

            #print("--Time step", timestep, "--")
            #environment.render()
            nActions = list()

            for agent in range(number_of_agents):
                action_i = learning_policy(state[agent], action_space, Q_functions[agent], epsilon, mode, weights)
                nActions.append(action_i)


            nObservations, nRewards, nDone, nInfo = environment.step(nActions)

            #print("Reward : ", nRewards[0])


            next_state = []

            for agent in range(number_of_agents):
                next_state.append(new_state(agent, nObservations[agent], tabularRL))
                Q_functions[agent] = update_Q(Q_functions[agent], state[agent], nActions[agent], nRewards[agent], next_state[agent], learning_rate, discount_factor, action_space, mode, weights)

            state = next_state

            if all(nDone):
                break

    return Q_functions


if __name__ == '__main__':
    tabularRL = True
    mode = "lex"
    weights = [1.0, 0.0]

    env = gym.make('CommonsGame:CommonsGame-v0', numAgents=number_of_agents, mapSketch=tinyMap, visualRadius=3,
                   fullState=False, tabularState=tabularRL, agent_pos=[[4, 3], [2, 0]])
    total_action_space = [i for i in range(env.action_space.n)]
    action_space = new_action_space(total_action_space, env)

    Qs = learning_loop(env, tabularRL, mode=mode)

    policy0 = policy_creator(Qs[0], action_space, mode=mode, weights=weights)
    policy1 = policy_creator(Qs[1], action_space, mode=mode, weights=weights)

    evaluation(env, tabularRL, policies=[policy0, policy1])
