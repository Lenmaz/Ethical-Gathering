import numpy as np
import gym

from constants import tinyMap, smallMap, bigMap, TOO_MANY_APPLES
from new_utils import *


policy_NULL = np.load("policy_NULL.npy")


def probsV_no_apples_in_ground(agent, V, nObservations, tabularRL, forced_agent_apples=-1):
    next_state = new_state(agent, nObservations[agent], tabularRL, forced_agent_apples)
    next_state_1 = new_state(agent, nObservations[agent],  tabularRL, forced_agent_apples, forced_grass=[True, False, False])
    next_state_2 = new_state(agent, nObservations[agent],  tabularRL, forced_agent_apples, forced_grass=[False, True, False])
    next_state_3 = new_state(agent, nObservations[agent],  tabularRL, forced_agent_apples, forced_grass=[False, False, True])

    return 0.85 * V[next_state] + 0.05 * V[next_state_1] + 0.05 * V[next_state_2] + 0.05 * V[next_state_3]


def probsV_calculator(agent, V, original_state, nObservations, tabularRL):
    checks_agent_before = check_agent_apples_state(agent, original_state[agent])
    checks_agent_after = check_agent_apples_state(agent, nObservations[agent])
    checks = check_apples_state(nObservations[agent])

    next_state = new_state(agent, nObservations[agent], tabularRL)

    # TODO: Añadir probabilidades acerca de agents apples (falta para n=4)

    if checks[0] or checks[1] or checks[2]: # Si no estARÉ en s' = no quedan manzanas

        if 0 < checks_agent_before < TOO_MANY_APPLES: # Pero estABA en el estado de s = tengo pocas manzanas

            if checks_agent_before < checks_agent_after:

                # HAS OBTENIDO UNA MANZANA
                next_state_A = new_state(agent, nObservations[agent], tabularRL, forced_agent_apples=1)
                next_state_B = new_state(agent, nObservations[agent], tabularRL, forced_agent_apples=2)  # Estado de transicion

                return 0.90 * V[next_state_A] + 0.10 * V[next_state_B]

            elif checks_agent_before > checks_agent_after:
                # HAS DONADO UNA MANZANA
                next_state_A = new_state(agent, nObservations[agent], tabularRL, forced_agent_apples=1)
                next_state_B = new_state(agent, nObservations[agent], tabularRL, forced_agent_apples=0)  # Estado de 0 manzanas

                return 0.90 * V[next_state_A] + 0.10 * V[next_state_B]

        return V[next_state]

    else:  # Si estARÉ en s' = no quedan manzanas

        if 0 < checks_agent_before < TOO_MANY_APPLES: # Y además estABA en el estado de s = tengo pocas manzanas

            if checks_agent_before < checks_agent_after:
                # HAS OBTENIDO UNA MANZANA
                probsV_A = probsV_no_apples_in_ground(agent, V, nObservations, tabularRL, 1)
                probsV_B = probsV_no_apples_in_ground(agent, V, nObservations, tabularRL, 2)  # Estado de transición
                return 0.90 * probsV_A + 0.10 * probsV_B

            elif checks_agent_before > checks_agent_after:
                # HAS DONADO UNA MANZANA
                probsV_A = probsV_no_apples_in_ground(agent, V, nObservations, tabularRL, 1)
                probsV_B = probsV_no_apples_in_ground(agent, V, nObservations, tabularRL, 0)  # Estado de 0 manzanas
                return 0.90 * probsV_A + 0.10 * probsV_B

        return probsV_no_apples_in_ground(agent, V, nObservations, tabularRL)



def scalarisation_function(values, w):
    """
    Scalarises the value of a state using a linear scalarisation function

    :param values: the different components V_0(s), ..., V_n(s) of the value of the state
    :param w:  the weight vector of the scalarisation function
    :return:  V(s), the scalarised value of the state
    """

    f = 0
    for objective in range(len(values)):
        f += w[objective]*values[objective]

    return f


def scalarised_Qs(len_action_space, Q_state, w):
    """
    Scalarises the value of each Q(s,a) for a given state using a linear scalarisation function

    :param Q_state: the different Q(s,a) for the state s, each with several components
    :param w: the weight vector of the scalarisation function
    :return: the scalarised value of each Q(s,a)
    """

    scalarised_Q = np.zeros(len_action_space)
    for action in range(len(Q_state)):
        scalarised_Q[action] = scalarisation_function(Q_state[action], w)

    return scalarised_Q


def lexicographic_Qs(action_space, Q_state):

    chosen_action = -1

    best_ethical_Q = np.max(scalarised_Qs(len(action_space), Q_state, only_ethical_matters))
    best_individual_Q = -np.inf

    for action in range(len(action_space)):
        q_Individual = scalarisation_function(Q_state[action], only_individual_matters)
        q_Ethical = scalarisation_function(Q_state[action], only_ethical_matters)
        if q_Ethical == best_ethical_Q:
            if q_Individual > best_individual_Q:
                best_individual_Q = q_Individual
                chosen_action = action

    #print(chosen_action, best_individual_Q, best_ethical_Q, Q_state)
    #print("----")

    return best_individual_Q , best_ethical_Q, action_space[chosen_action]


def evaluation(agent, env, policy, tabularRL, learning_agents=2):

    initial_state = env.reset()

    states = list()
    for ag in range(number_of_agents):
        states.append(new_state(ag, initial_state[agent], tabularRL))


    for t in range(500):

        env.render()

        actions = [policy_NULL[states[0]], policy_NULL[states[1]]]
        actions[agent] = policy[states[agent]]

        print(agent, states[agent], policy[states[agent]])

        if learning_agents == 2:
            #actions[agent - 1 % 2] = policy[states[agent - 1 % 2]]
            pass
            #TODO: Correct

        nObservations, rewards, nDone, _ = env.step(actions)

        states = list()
        for ag in range(number_of_agents):
            states.append(new_state(ag, initial_state[agent], tabularRL))

        print("--Time step", t, "--")



def policy_creator(Q_function, action_space, mode="scalarisation", weights=[1.0 , 0.0]):

    policy = np.zeros(len_state_space)

    for state in range(len_state_space):

        if mode == "scalarisation":
            index = np.argmax(scalarised_Qs(len(action_space), Q_function[state][action_space], weights))
            policy[state] = action_space[index]
        elif mode == "lex":
            _, _, policy[state] = lexicographic_Qs(action_space, Q_function[state][action_space])
        else:
            policy[state] = 6  # env.STAY


    return policy



def sweep_Q_function(agent, Q, V, action_space, mode, discount_factor, weights):
    """
    TODO: Adapt
    Calculates the value of applying each action to a given state. Heavily adapted to the public civility game

    :param env: the environment of the Markov Decision Process
    :param state: the current state
    :param V: value function to see the value of the next state V(s')
    :param discount_factor: discount factor considered, a real number
    :return: the value obtained for each action
    """

    state_count = 0

    for ag_pos in agent_positions:

        for ap_state in apple_states:

            if check_redundant_states(positions_with_apples, ag_pos, ap_state):
                continue

            for n_apples in agent_num_apples:

                for c_state in common_pool_states:

                    state_count += 1

                    print("States swept :", state_count, "/", len_state_space + 1)


                    for action in action_space:

                        env = gym.make('CommonsGame:CommonsGame-v0', numAgents=number_of_agents, mapSketch=tinyMap,
                                       visualRadius=3, fullState=False, tabularState=tabularRL, agent_pos=ag_pos)
                        original_state = env.reset(num_apples=n_apples, common_pool=c_state, apples_yes_or_not=ap_state)
                        state = new_state(agent, original_state[agent], tabularRL)

                        actions = [policy_NULL[state], policy_NULL[state]]
                        actions[agent] = action

                        nObservations, nRewards, _, _ = env.step(actions)

                        #for agent in range(number_of_agents):

                        reward = nRewards[agent]

                        V_prima = probsV_calculator(agent, V[agent], original_state, nObservations, tabularRL)

                        Q[agent][state][action] = reward + discount_factor * V_prima


                    # Update V
                    #for agent in range(number_of_agents):
                    if mode == "lex":
                        V_ind, V_eth, _ = lexicographic_Qs(action_space, Q[agent][state][action_space])
                        V[agent][state] = np.array([V_ind, V_eth])
                    elif mode == "scalarisation":
                        index = np.argmax(scalarised_Qs(len(action_space), Q[agent][state][action_space], weights))
                        V[agent][state] = Q[agent][state][action_space[index]]
    return Q, V


def value_iteration(tabularRL, agent=who_is_the_learning_agent, mode="lex", discount_factor=0.8, weights=[1.0, 0.0], num_iterations=1):
    """
    Adapted for VI

    :param environment: the environment already configured
    :param tabularRL: boolean to know if you will be using tabular RL or deep RL
    :return:
    """

    environment = gym.make('CommonsGame:CommonsGame-v0', numAgents=number_of_agents, mapSketch=tinyMap, visualRadius=3, fullState=False, tabularState=tabularRL)

    total_action_space = [i for i in range(environment.action_space.n)]
    action_space = new_action_space(total_action_space, environment)

    Q_functions = np.zeros((number_of_agents, len_state_space, environment.action_space.n, number_of_objectives))
    V_functions = np.zeros((number_of_agents, len_state_space, number_of_objectives))
    for episode in range(num_iterations):
        print(episode)
        Q_functions, V_functions = sweep_Q_function(agent, Q_functions, V_functions, action_space, mode, discount_factor, weights)

    return Q_functions, V_functions


if __name__ == '__main__':
    tabularRL = True
    weights = [1.0, 1.0]
    mode = "lex"

    environment = gym.make('CommonsGame:CommonsGame-v0', numAgents=number_of_agents, mapSketch=tinyMap, visualRadius=3, fullState=False, tabularState=tabularRL)

    total_action_space = [i for i in range(environment.action_space.n)]
    action_space = new_action_space(total_action_space, environment)

    #Q_functions, V_functions = value_iteration(tabularRL, mode=mode, weights=weights)
    #np.save("Q_func.npy", Q_functions[0])
    #np.save("V_func.npy", V_functions[0])
    Q_function = np.load("Q_func.npy")
    policy = policy_creator(Q_function, action_space, mode=mode, weights=weights)
    np.save("policy.npy", policy)
    policy = np.load("policy.npy")


    #policy = policy_creator(None, None, mode="dumb")
    #np.save("policy_NULL.npy", policy)
    #policy = np.load("policy_NULL.npy")



    env = gym.make('CommonsGame:CommonsGame-v0', numAgents=number_of_agents, mapSketch=tinyMap, visualRadius=3, fullState=False, tabularState=tabularRL, agent_pos=[[4, 0], [2, 0]])
    evaluation(who_is_the_learning_agent, env, policy, tabularRL)

    #V = np.load("V_func.npy")
    #for pos in agent_positions:
    #    state = new_state(0, [0], True, forced_agent_apples=0, forced_grass=[True, True, True], forced_ag_pos=pos)
    #    print(pos)
    #    print(V[state])
    #    print("--------")