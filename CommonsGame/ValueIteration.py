import numpy as np
import gym

from constants import tinyMap, smallMap, bigMap, TOO_MANY_APPLES
from new_utils import *


def probsV_on_apples_in_ground(agent, V, state_prima, tabularRL, forced_agent_apples=-1, apples_in_ground=False, forced_pool=-1):
    """

    Computes the probability of moving from the current state to another state where
    at least there is one apple in ground. It also computes its associated V(s').

    If there are apples_in_ground, of course the probability is 100 %

    """
    next_state = new_state(agent, state_prima, tabularRL, forced_pool=forced_pool)

    if apples_in_ground:
        return V[next_state]
    else:
        next_state_1 = new_state(agent, state_prima,  tabularRL, forced_agent_apples, forced_grass=[True, False, False], forced_pool=forced_pool)
        next_state_2 = new_state(agent, state_prima,  tabularRL, forced_agent_apples, forced_grass=[False, True, False], forced_pool=forced_pool)
        next_state_3 = new_state(agent, state_prima,  tabularRL, forced_agent_apples, forced_grass=[False, False, True], forced_pool=forced_pool)

        return 0.85 * V[next_state] + 0.05 * V[next_state_1] + 0.05 * V[next_state_2] + 0.05 * V[next_state_3]


def probsV_on_agent_apples_auxiliar(agent, V, state_prima, tabularRL, forced_pool, apples_in_ground, agent_gains=True):

    """

    Auxiliar for the following method. This one is only used when in the transition,
    the real number of apples of the agent has changed.

    agent_gains = True if the real number of apples has increased, False if it has decreased

    Notice that the if the number of apples has not changed, this method should not even be called.

    """
    if agent_gains:
        next_agent_apples = 2
    else:
        next_agent_apples = 0

    p = 0.1

    probs = [1-p, p]
    agent_apples = [1, next_agent_apples]

    probsV_total = 0

    for i in range(len(probs)):
        probsV_total += probs[i]*probsV_on_apples_in_ground(agent, V, state_prima, tabularRL, forced_agent_apples=agent_apples[i],
                                          apples_in_ground=apples_in_ground, forced_pool=forced_pool)

    return probsV_total


def probsV_on_agent_apples(agent, V, original_state, state_prima, tabularRL, forced_pool_apples, apples_in_ground):
    """

    Computes the probability of moving from a state where the agent has several apples (but neither 0 nor TOO_MANY)
    from either a state where it has 0 apples (if the agent decides to donate)
    or a state where it has TOO_MANY_APPLES (if the agent collects an apple)

    and in either case it computes the corresponding V(s')

    If the agent is in a different apple_state, of course the computation is trivial.

    """
    checks_agent_before = check_agent_apples_state(agent, original_state)
    checks_agent_after = check_agent_apples_state(agent, state_prima)

    if 0 < checks_agent_before < TOO_MANY_APPLES:  # Y además estABA en el estado de s = tengo pocas manzanas

        if checks_agent_before != checks_agent_after:

            if checks_agent_before < checks_agent_after:  # +1 in apples
                agent_gains = True
            else:                                         # -1 in apples
                agent_gains = False

            return probsV_on_agent_apples_auxiliar(agent, V, state_prima, tabularRL, forced_pool=forced_pool_apples, apples_in_ground=apples_in_ground, agent_gains=agent_gains)

    return probsV_on_apples_in_ground(agent, V, state_prima, tabularRL, forced_pool=forced_pool_apples, apples_in_ground=apples_in_ground)


def probsV_apples_ground_and_agents(agent, V, original_state, state_prima, tabularRL, forced_pool_apples=-1):
    """

    Fusion of the two previous methods. This one actually checks whether we are in a no-apples-in-ground state
    and then redirects to the corresponding method that needs to be applied.

    """
    checks = check_apples_state(original_state)
    there_are_apples_in_ground = checks[0] or checks[1] or checks[2] # True or False

    return probsV_on_agent_apples(agent, V, original_state, state_prima, tabularRL, forced_pool_apples=forced_pool_apples, apples_in_ground=there_are_apples_in_ground)


def probsV_calculator(agent, action, V, original_state, state_prima, tabularRL):
    """

    We include in the previous method the checking of whether the common pool is in the state 2
    (i.e., more than 1 apple, but less than 24 apples) and we also check if the agents
    have decided to donate to the common pool.

    We compute the probability of the next state and its associated V(s').
    """
    apples_in_pool = check_common_pool(original_state)

    if apples_in_pool == 2:
        if action == 8:
            # There are between 24 and two apples in the pool and the agent donated an apple
            # So 95% chance that the next state is the same
            p = 1.0 / (TOO_MANY_APPLES * number_of_agents - 2.0)

            probs = [1 - p, p]
            pool_apples = [2, TOO_MANY_APPLES * number_of_agents]

            probsV_total = 0

            for i in range(len(probs)):
                probsV_total += probs[i] * probsV_apples_ground_and_agents(agent, V, original_state, state_prima, tabularRL,
                                                                           forced_pool_apples=pool_apples[i])

            return probsV_total
        elif action == 9:
            # There are between 24 and two apples in the pool and the agent took an apple from the pool
            # So 95% chance that the next state is the same
            p = 1.0 / (TOO_MANY_APPLES * number_of_agents - 2.0)

            probs = [1 - p, p]
            pool_apples = [2, 1]

            probsV_total = 0

            for i in range(len(probs)):
                probsV_total += probs[i] * probsV_apples_ground_and_agents(agent, V, original_state, state_prima, tabularRL,
                                                                           forced_pool_apples=pool_apples[i])

            return probsV_total

    elif apples_in_pool == TOO_MANY_APPLES * number_of_agents:
        if action == 9:

            # There are 24 apples or more in the pool and the agent took an apple from the pool
            # So 95% chance that the next state is the same
            p = 1.0 / (TOO_MANY_APPLES * number_of_agents - 2.0)

            probs = [1 - p, p]
            pool_apples = [TOO_MANY_APPLES * number_of_agents, 2]

            probsV_total = 0

            for i in range(len(probs)):
                probsV_total += probs[i] * probsV_apples_ground_and_agents(agent, V, original_state, state_prima, tabularRL,
                                                                           forced_pool_apples=pool_apples[i])

            return probsV_total

    return probsV_apples_ground_and_agents(agent, V, original_state, state_prima, tabularRL)



def create_model(agent):

    state_count = 0

    model = {}

    for ag_pos in agent_positions:

        for ap_state in apple_states:

            if check_redundant_states(positions_with_apples, ag_pos, ap_state):
                continue

            for n_apples in agent_num_apples:

                for c_state in common_pool_states:

                    state_count += 1

                    env = gym.make('CommonsGame:CommonsGame-v0', numAgents=number_of_agents, mapSketch=tinyMap,
                                   visualRadius=3, fullState=False, tabularState=tabularRL, agent_pos=ag_pos)
                    original_state = env.reset(num_apples=n_apples, common_pool=c_state, apples_yes_or_not=ap_state)

                    model[original_state[agent]] = {}


                    for action in action_space:
                        env = gym.make('CommonsGame:CommonsGame-v0', numAgents=number_of_agents, mapSketch=tinyMap,
                                       visualRadius=3, fullState=False, tabularState=tabularRL, agent_pos=ag_pos)
                        original_state = env.reset(num_apples=n_apples, common_pool=c_state, apples_yes_or_not=ap_state)

                        state = new_state(agent, original_state[agent], tabularRL)

                        actions = [policy_0[state], policy_1[state]]
                        actions[agent] = action

                        nObservations, nRewards, _, _ = env.step(actions)

                        model[original_state][action] = [nObservations[agent], nRewards[agent]]

    return model


def sweep_Q_function_with_model(agent, model, Q, V, action_space, mode, discount_factor, weights):
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

    for original_state in model:

        state_count += 1

        if state_count % 100 == 0:
            print("States swept :", state_count, "/", len_state_space + 1)

        for action in model[original_state]:

                state = new_state(agent, original_state[agent], tabularRL)
                state_prima, reward = model[original_state][action]


                V_prima = probsV_calculator(agent, action, V[agent], original_state, state_prima, tabularRL)

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

                    if state_count % 100 == 0:
                        print("States swept :", state_count, "/", len_state_space + 1)


                    for action in action_space:

                        env = gym.make('CommonsGame:CommonsGame-v0', numAgents=number_of_agents, mapSketch=tinyMap,
                                       visualRadius=3, fullState=False, tabularState=tabularRL, agent_pos=ag_pos)
                        original_state = env.reset(num_apples=n_apples, common_pool=c_state, apples_yes_or_not=ap_state)
                        state = new_state(agent, original_state[agent], tabularRL)

                        actions = [policy_0[state], policy_1[state]]
                        actions[agent] = action

                        nObservations, nRewards, _, _ = env.step(actions)



                        #for agent in range(number_of_agents):

                        reward = nRewards[agent]

                        V_prima = probsV_calculator(agent, actions[agent], V[agent], original_state[agent], nObservations[agent], tabularRL)

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


def value_iteration(tabularRL, agent, mode="lex", discount_factor=0.8, weights=[1.0, 0.0], num_iterations=3):
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
    weights = [1.0, 0.0]
    mode = "lex"

    environment = gym.make('CommonsGame:CommonsGame-v0', numAgents=number_of_agents, mapSketch=tinyMap, visualRadius=3, fullState=False, tabularState=tabularRL)

    total_action_space = [i for i in range(environment.action_space.n)]
    action_space = new_action_space(total_action_space, environment)

    #Q_functions, V_functions = value_iteration(tabularRL, who_is_the_learning_agent, mode=mode, weights=weights)
    #np.save("Q_func.npy", Q_functions[who_is_the_learning_agent])
    #np.save("V_func.npy", V_functions[who_is_the_learning_agent])
    #Q_function = np.load("Q_func.npy")
    #policy = policy_creator(Q_function, action_space, mode=mode, weights=weights)
    #np.save("policy_"+str(who_is_the_learning_agent)+".npy", policy)

    policy = np.load("policy_NULL.npy")

    env = gym.make('CommonsGame:CommonsGame-v0', numAgents=number_of_agents, mapSketch=tinyMap, visualRadius=3, fullState=False, tabularState=tabularRL, agent_pos=[[2, 0],[4, 0]])
    evaluation(env, tabularRL)

    #V = np.load("V_func.npy")
    #for pos in agent_positions:
    #    state = new_state(0, [0], True, forced_agent_apples=0, forced_grass=[True, True, True], forced_ag_pos=pos)
    #    print(pos)
    #    print(V[state])
    #    print("--------")