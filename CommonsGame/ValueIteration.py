import numpy as np
import gym

from constants import tinyMap, smallMap, bigMap, TOO_MANY_APPLES, training_now, COMMON_POOL_HAS_LIMIT
from new_utils import *


def probsV_on_apples_in_ground(agent, V, state_prima, tabularRL, forced_agent_apples=-1, apples_in_ground=0, forced_pool=-1, original_state=-1):
    """

    Computes the probability of moving from the current state to another state where
    at least there is one apple in ground. It also computes its associated V(s').

    If there are apples_in_ground, of course the probability is 100 %

    """
    next_state = new_state(agent, state_prima, tabularRL, forced_agent_apples, forced_pool=forced_pool)

    #try:
    #    if new_state(0, original_state, tabularRL) == 5992:
    #        print("heyyyy")
    #        print(next_state)
    #except:
    #    pass
    where_apples, where_agents = apples_in_ground
    there_are_apples_in_ground = where_apples[0] or where_apples[1] or where_apples[2] # True or False
    agents_are_where_apples = where_agents[0] or where_agents[1] or where_agents[2]

    if there_are_apples_in_ground:
        return V[next_state]
    else:
        next_state_1 = new_state(agent, state_prima,  tabularRL, forced_agent_apples, forced_grass=[not where_agents[0], False, False], forced_pool=forced_pool)
        next_state_2 = new_state(agent, state_prima,  tabularRL, forced_agent_apples, forced_grass=[False, not where_agents[1], False], forced_pool=forced_pool)
        next_state_3 = new_state(agent, state_prima,  tabularRL, forced_agent_apples, forced_grass=[False, False, not where_agents[2]], forced_pool=forced_pool)

        return 0.85 * V[next_state] + 0.05 * V[next_state_1] + 0.05 * V[next_state_2] + 0.05 * V[next_state_3]


def probsV_on_agent_apples_auxiliar(agent, V, state_prima, tabularRL, forced_pool, apples_in_ground, agent_gains=True, forcing_apples=False, original_state=-1):

    """

    Auxiliar for the following method. This one is only used when in the transition,
    the real number of apples of the agent has changed.

    agent_gains = True if the real number of apples has increased, False if it has decreased

    Notice that the if the number of apples has not changed, this method should not even be called.

    """
    if agent_gains:
        next_agent_apples = TOO_MANY_APPLES
    else:
        next_agent_apples = 0

    p = 0.1

    if forcing_apples:
        p = 0.05

    probs = [1-p, p]
    agent_apples = [TOO_MANY_APPLES - 1, next_agent_apples]

    probsV_total = 0

    for i in range(len(probs)):
        addition = probs[i]*probsV_on_apples_in_ground(agent, V, state_prima, tabularRL, forced_agent_apples=agent_apples[i],
                                          apples_in_ground=apples_in_ground, forced_pool=forced_pool, original_state=original_state)

        probsV_total += addition
        #if new_state(0, original_state, True) == 5992:
        #    print(probs[i], agent_apples[i], addition)

    return probsV_total


def probsV_on_agent_apples(agent, V, original_state, state_prima, tabularRL, forced_pool_apples, apples_in_ground, forcing_next_apples=False):
    """

    Computes the probability of moving from a state where the agent has several apples (but neither 0 nor TOO_MANY)
    from either a state where it has 0 apples (if the agent decides to donate)
    or a state where it has TOO_MANY_APPLES (if the agent collects an apple)

    and in either case it computes the corresponding V(s')

    If the agent is in a different apple_state, of course the computation is trivial.

    """
    checks_agent_before = check_agent_apples_state(agent, original_state)

    if forcing_next_apples:

        V_A = probsV_on_apples_in_ground(agent, V, state_prima, tabularRL, forced_agent_apples=checks_agent_before,
                                          apples_in_ground=apples_in_ground, forced_pool=forced_pool_apples)

        if checks_agent_before != TOO_MANY_APPLES - 1:
            V_B = probsV_on_apples_in_ground(agent, V, state_prima, tabularRL, forced_agent_apples=checks_agent_before+1,
                                             apples_in_ground=apples_in_ground, forced_pool=forced_pool_apples)
        else:
            V_B = probsV_on_agent_apples_auxiliar(agent, V, state_prima, tabularRL, forced_pool=forced_pool_apples, apples_in_ground=apples_in_ground, agent_gains=True, forcing_apples=True, original_state=original_state)
        return 0.5* V_A + 0.5 * V_B


    checks_agent_after = check_agent_apples_state(agent, state_prima)

    if 0 < checks_agent_before < TOO_MANY_APPLES:  # Y además estABA en el estado de s = tengo pocas manzanas

        if checks_agent_before != checks_agent_after:

            if checks_agent_before < checks_agent_after:  # +1 in apples
                agent_gains = True
            else:                                         # -1 in apples
                agent_gains = False

            return probsV_on_agent_apples_auxiliar(agent, V, state_prima, tabularRL, forced_pool=forced_pool_apples, apples_in_ground=apples_in_ground, agent_gains=agent_gains, original_state=original_state)

    return probsV_on_apples_in_ground(agent, V, state_prima, tabularRL, forced_pool=forced_pool_apples, apples_in_ground=apples_in_ground)


def probsV_apples_ground_and_agents(agent, V, original_state, state_prima, tabularRL, forced_pool_apples=-1, forcing_next_apples=False):
    """

    Fusion of the two previous methods. This one actually checks whether we are in a no-apples-in-ground state
    and then redirects to the corresponding method that needs to be applied.

    """
    checks1 = check_apples_state(original_state)
    checks2 = check_agents_where_apples(original_state)

    return probsV_on_agent_apples(agent, V, original_state, state_prima, tabularRL, forced_pool_apples=forced_pool_apples, apples_in_ground=[checks1, checks2], forcing_next_apples=forcing_next_apples)


def probsV_calculator(agent, action, V, original_state, state_prima, tabularRL, forcing_next_apples=False):
    """

    We include in the previous method the checking of whether the common pool is in the state 2
    (i.e., more than 1 apple, but less than 24 apples) and we also check if the agents
    have decided to donate to the common pool.

    We compute the probability of the next state and its associated V(s').
    """

    if COMMON_POOL_HAS_LIMIT:
        apples_in_pool = check_common_pool(original_state)

        if apples_in_pool == 2:
            if action == 8:
                # There are between 24 and two apples in the pool and the agent donated an apple
                # So 95% chance that the next state is the same
                p = 1.0 / (common_pool_states[-1] - 2.0)

                probs = [1 - p, p]
                pool_apples = [2, common_pool_states[-1]]

                probsV_total = 0

                for i in range(len(probs)):
                    probsV_total += probs[i] * probsV_apples_ground_and_agents(agent, V, original_state, state_prima, tabularRL,
                                                                               forced_pool_apples=pool_apples[i])

                return probsV_total
            elif action == 9:
                # There are between 24 and two apples in the pool and the agent took an apple from the pool
                # So 95% chance that the next state is the same
                p = 1.0 / (common_pool_states[-1] - 2.0)

                probs = [1 - p, p]
                pool_apples = [2, 1]

                probsV_total = 0

                for i in range(len(probs)):
                    probsV_total += probs[i] * probsV_apples_ground_and_agents(agent, V, original_state, state_prima, tabularRL,
                                                                               forced_pool_apples=pool_apples[i])

                return probsV_total

        #TODO: Decide what to do with this
        elif False: #apples_in_pool == TOO_MANY_APPLES * number_of_agents:
            if action == 9:

                # There are 24 apples or more in the pool and the agent took an apple from the pool
                # So 95% chance that the next state is the same
                p = 1.0 / (common_pool_states[-1] - 2.0)

                probs = [1 - p, p]
                pool_apples = [common_pool_states[-1], 2]

                probsV_total = 0

                for i in range(len(probs)):
                    probsV_total += probs[i] * probsV_apples_ground_and_agents(agent, V, original_state, state_prima, tabularRL,
                                                                               forced_pool_apples=pool_apples[i])

                return probsV_total

    return probsV_apples_ground_and_agents(agent, V, original_state, state_prima, tabularRL, forcing_next_apples=forcing_next_apples)




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

                        if not check_random_reward(original_state[agent], actions):
                            V_prima = probsV_calculator(agent, actions[agent], V[agent], original_state[agent], nObservations[agent], tabularRL)

                            Q[agent][state][action] = reward + discount_factor * V_prima
                        else:

                            V_prima = probsV_calculator(agent, actions[agent], V[agent], original_state[agent],
                                                        nObservations[agent], tabularRL, forcing_next_apples=True)

                            Q[agent][state][action] = np.array([-0.5, reward[1]]) + discount_factor * V_prima

                        if state == 9687:
                            print(state, original_state[agent], new_state(1, nObservations[agent], True), actions, reward, V_prima, nObservations[agent])


                    if state == 9687:
                        print("conflictive state hmm", original_state[agent], Q[agent][state])
                        print("---")


    # Update V
    #for agent in range(number_of_agents):
    for state in range(len_state_space):
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

    if training_now:
        for iteration in [0,1,2]:

            if iteration == 0:
                policy_0 = np.load("policy_NULL.npy")
                policy_1 = np.load("policy_NULL.npy")
            else:
                policy_0 = np.load("policy_0_i" + str(iteration - 1) + ".npy")
                policy_1 = np.load("policy_1_i" + str(iteration - 1) + ".npy")

            for learner in [0, 1]:



                Q_functions, V_functions = value_iteration(tabularRL, learner, mode=mode, weights=weights)
                np.save("Q_func.npy", Q_functions[learner])
                np.save("V_func.npy", V_functions[learner])
                Q_function = np.load("Q_func.npy")
                policy = policy_creator(Q_function, action_space, mode=mode, weights=weights)
                np.save("policy_"+str(learner)+"_i"+str(iteration)+".npy", policy)

    policy = np.load("policy_NULL.npy")

    env = gym.make('CommonsGame:CommonsGame-v0', numAgents=number_of_agents, mapSketch=tinyMap, visualRadius=3, fullState=False, tabularState=tabularRL, agent_pos=[[2, 0],[4, 0]])
    evaluation(env, tabularRL)

    #V = np.load("V_func.npy")
    #for pos in agent_positions:
    #    state = new_state(0, [0], True, forced_agent_apples=0, forced_grass=[True, True, True], forced_ag_pos=pos)
    #    print(pos)
    #    print(V[state])
    #    print("--------")