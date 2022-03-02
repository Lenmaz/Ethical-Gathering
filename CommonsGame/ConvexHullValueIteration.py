import numpy as np
import gym
import convexhull


from constants import tinyMap, smallMap, bigMap, TOO_MANY_APPLES
from new_utils import *

policy_NULL = np.load("policy_NULL.npy")
policy_0 = np.load("policy_NULL.npy")
policy_1 = np.load("policy_NULL.npy")

def probsV_no_apples_in_ground(agent, reward, tabularRL, discount_factor, V, nObservations, forced_agent_apples=-1, corrector=1.0):

    next_state = new_state(agent, nObservations[agent], tabularRL, forced_agent_apples)
    next_state_1 = new_state(agent, nObservations[agent], tabularRL, forced_agent_apples, forced_grass=[True, False, False])
    next_state_2 = new_state(agent, nObservations[agent], tabularRL, forced_agent_apples, forced_grass=[False, True, False])
    next_state_3 = new_state(agent, nObservations[agent], tabularRL, forced_agent_apples, forced_grass=[False, False, True])

    hull_sa = convexhull.translate_hull(corrector * 0.85 * reward, discount_factor, corrector * 0.85 * V[next_state].copy())
    hull_sa1 = convexhull.translate_hull(corrector * 0.05 * reward, discount_factor, corrector * 0.05 * V[next_state_1].copy())
    hull_sa2 = convexhull.translate_hull(corrector * 0.05 * reward, discount_factor, corrector * 0.05 * V[next_state_2].copy())
    hull_sa3 = convexhull.translate_hull(corrector * 0.05 * reward, discount_factor, corrector * 0.05 * V[next_state_3].copy())

    hull_sa = convexhull.sum_hulls(hull_sa, hull_sa1)
    hull_sa = convexhull.sum_hulls(hull_sa, hull_sa2)
    hull_sa = convexhull.sum_hulls(hull_sa, hull_sa3)

    #print("---AAAAA-----")
    #print(next_state, V[next_state], V[next_state_1], V[next_state_2], V[next_state_3])
    #print("----BBBBB-----")
    return hull_sa

def probsV_calculator(agent, reward, tabularRL, discount_factor, V, original_state, nObservations):
    checks_agent_before = check_agent_apples_state(agent, original_state[agent])
    checks_agent_after = check_agent_apples_state(agent, nObservations[agent])
    checks = check_apples_state(nObservations[agent])

    next_state = new_state(agent, nObservations[agent], tabularRL)

    # TODO: Añadir probabilidades acerca de agents apples (falta para n=4)
    # TODO: Añadir probabilidades acerca de common pool (puedes esperar al multiagent)

    if checks[0] or checks[1] or checks[2]: # Si no estARÉ en s' = no quedan manzanas

        if 0 < checks_agent_before < TOO_MANY_APPLES: # Pero estABA en el estado de s = tengo pocas manzanas

            if checks_agent_before < checks_agent_after:
                # HAS OBTENIDO UNA MANZANA
                next_state_A = new_state(agent, nObservations[agent], tabularRL, forced_agent_apples=1)
                next_state_B = new_state(agent, nObservations[agent], tabularRL, forced_agent_apples=2)  # Estado de transicion

                hull_saA = convexhull.translate_hull(0.9 * reward, discount_factor, 0.9 * V[next_state_A].copy())
                hull_saB = convexhull.translate_hull(0.1 * reward, discount_factor, 0.1 * V[next_state_B].copy())

                return convexhull.sum_hulls(hull_saA, hull_saB)

            elif checks_agent_before > checks_agent_after:
                # HAS DONADO UNA MANZANA

                next_state_A = new_state(agent, nObservations[agent], tabularRL, forced_agent_apples=1)
                next_state_B = new_state(agent, nObservations[agent], tabularRL, forced_agent_apples=0)  # Estado de 0 manzanas

                hull_saA = convexhull.translate_hull(0.9 * reward, discount_factor, 0.9 * V[next_state_A].copy())
                hull_saB = convexhull.translate_hull(0.1 * reward, discount_factor, 0.1 * V[next_state_B].copy())

                return convexhull.sum_hulls(hull_saA, hull_saB)

        return convexhull.translate_hull(reward, discount_factor, V[next_state].copy())

    else:  # Si estARÉ en s' = no quedan manzanas

        if 0 < checks_agent_before < TOO_MANY_APPLES: # Y además estABA en el estado de s = tengo pocas manzanas

            if checks_agent_before < checks_agent_after:
                # HAS OBTENIDO UNA MANZANA
                probsV_A = probsV_no_apples_in_ground(agent, reward, tabularRL, discount_factor, V, nObservations, 1, 0.9)
                probsV_B = probsV_no_apples_in_ground(agent, reward, tabularRL, discount_factor, V, nObservations, 2, 0.1)  # Estado de transición
                return convexhull.sum_hulls(probsV_A, probsV_B)

            elif checks_agent_before > checks_agent_after:
                # HAS DONADO UNA MANZANA
                #print("YEP")
                probsV_A = probsV_no_apples_in_ground(agent, reward, tabularRL, discount_factor, V, nObservations, 1, 0.9)
                probsV_B = probsV_no_apples_in_ground(agent, reward, tabularRL, discount_factor, V, nObservations, 0, 0.1)  # Estado de 0 manzanas
                #print(probsV_A, probsV_B)
                return convexhull.sum_hulls(probsV_A, probsV_B)

        return probsV_no_apples_in_ground(agent, reward, tabularRL, discount_factor, V, nObservations)



def sweep_Q_function(agent, tabularRL, V, action_space, discount_factor, tolerance=1.0):
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

                    print("States swept :", state_count, "/", len_state_space+1)

                    hulls = list()

                    for nActions in action_space:
                        env = gym.make('CommonsGame:CommonsGame-v0', numAgents=number_of_agents, mapSketch=tinyMap,
                                       visualRadius=3, fullState=False, tabularState=tabularRL, agent_pos=ag_pos)
                        original_state = env.reset(num_apples=n_apples, common_pool=c_state, apples_yes_or_not=ap_state)

                        state = new_state(agent, original_state[agent], tabularRL)

                        actions = [policy_0[state], policy_1[state]]
                        actions[agent] = nActions

                        nObservations, nRewards, _, _ = env.step(actions)


                        reward = np.array(nRewards[agent])

                        hull_sa = probsV_calculator(agent, reward, tabularRL, discount_factor, V, original_state, nObservations)

                        #if state_count == 4:
                        #    print(reward, hull_sa)

                        for point in hull_sa:
                            hulls.append(point)

                    hulls = np.unique(np.array(hulls), axis=0)
                    #V[state] = convexhull.get_hull(hulls)
                    V_state_unprocessed = convexhull.get_hull(hulls)
                    V[state] = convexhull.check_descent(V_state_unprocessed, tolerance=tolerance)


                    #print(V[state])
    #from SAEEP import ethical_embedding_state

    #print( V[346], ethical_embedding_state(V[346], tolerance=1.0))
    return V


def CH_value_iteration(agent, tabularRL, discount_factor=0.8, num_iterations=3, tolerance=1.0):
    """
    Adapted for VI

    :param environment: the environment already configured
    :param tabularRL: boolean to know if you will be using tabular RL or deep RL
    :return:
    """

    environment = gym.make('CommonsGame:CommonsGame-v0', numAgents=number_of_agents, mapSketch=tinyMap, visualRadius=3, fullState=False, tabularState=tabularRL)

    total_action_space = [i for i in range(environment.action_space.n)]
    action_space = new_action_space(total_action_space, environment)

    V = list()
    for i in range(number_of_agents):
        V.append(list())
        for _ in range(len_state_space):
            V[i].append(np.array([]))

    for episode in range(num_iterations):
        print(episode)
        V[agent] = sweep_Q_function(agent, tabularRL, V[agent], action_space, discount_factor, tolerance)

    return V[agent]  #TODO: Un poco feo esto


if __name__ == '__main__':
    tabularRL = True

    V_functions = CH_value_iteration(who_is_the_learning_agent, tabularRL, tolerance=0.8)

    num_states = len(V_functions)

    V = V_functions

