import numpy as np
from constants import TOO_MANY_APPLES, COMMON_POOL_HAS_LIMIT


policy_NULL = np.load("policy_NULL.npy")
#policy_0 = np.load("policy_0_i" + str(3) + ".npy")
#policy_1 = np.load("policy_1_i" + str(3) + ".npy")

number_of_agents = 2
#who_is_the_learning_agent = 0
number_of_objectives = 2

only_ethical_matters = [0.0, 1.0]
only_individual_matters = [1.0, 0.0]

agent_positions = list()

# for tinyMap
agent_x_position = 3
agent_y_position = 4

for x in range(2, 2 + agent_x_position):
    for y in range(agent_y_position):
        agent_positions.append([x, y])


two_agent_positions = list()

for ag_pos in agent_positions:
    for ag_pos2 in agent_positions:
        two_agent_positions.append([ag_pos, ag_pos2])


if number_of_agents == 2:
    agent_positions = two_agent_positions

# for tinyMap
positions_with_apples = [[2, 1], [2, 2], [3, 1]]
agent_num_apples = [0, TOO_MANY_APPLES - 1, TOO_MANY_APPLES, TOO_MANY_APPLES + 1]
if COMMON_POOL_HAS_LIMIT:
    common_pool_states = [0, 1, 2, TOO_MANY_APPLES*number_of_agents]
else:
    common_pool_states = [0, 1, 2]

apple_states = list()  # for this to work you need SUSTAINABILITY_MATTERS TRUE, which makes that no new apple appears
for i in range(2):
    for j in range(2):
        for k in range(2):
            apple_states.append([bool(i), bool(j), bool(k)])


n_agent_cells = len(agent_positions) if number_of_agents == 1 else len(agent_positions)
n_apples = len(agent_num_apples)
apples_in_ground = len(apple_states)
common_pool_max = len(common_pool_states)

len_state_space = n_agent_cells*n_apples*apples_in_ground*common_pool_max + 1  # You need to change this!!! This is provisional and only works for tinyMap


print("State space: ", len_state_space)

def new_action_space(action_space, env):
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


def new_state(agent, state, tabularRL, forced_agent_apples=-1, forced_grass=[False, False, False], forced_ag_pos=[], forced_pool=-1):
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
            return len_state_space
        else:
            # Obtain the agent's position:

            agents_x = list()
            agents_y = list()

            if len(forced_ag_pos) > 0:

                for ag in range(number_of_agents):
                    agents_x.append(forced_ag_pos[0])
                    agents_y.append(forced_ag_pos[1])
            else:
                for ag in range(number_of_agents):
                    agents_x.append(state[-1 - 4 * number_of_agents + 4 * ag])
                    agents_y.append(state[-4 * number_of_agents + 4 * ag])

            position = 0

            for ag in range(number_of_agents):
                position += (agents_x[ag] + agent_x_position*agents_y[ag])*(agent_x_position*agent_y_position)**ag   # we encode them as a scalar, there are 16 different positions

            # Obtain the agent's amount of apples, but we only consider four possible values
            if forced_agent_apples > -1:
                agent_temp_apples = forced_agent_apples
            else:
                agent_temp_apples = state[1 - 4 * number_of_agents + 4 * agent]
            if agent_temp_apples == 0:
                agent_apples = 0
            elif agent_temp_apples < TOO_MANY_APPLES:
                agent_apples = 1
            elif agent_temp_apples == TOO_MANY_APPLES:
                agent_apples = 2
            else:
                agent_apples = 3

            if forced_pool >= 0:

                if forced_pool < common_pool_states[-1] or not COMMON_POOL_HAS_LIMIT:
                    common_pool_apples = min(forced_pool, 2)
                else:
                    common_pool_apples = 3
            else:
                real_common_pool = state[-1]

                if real_common_pool < common_pool_states[-1] or not COMMON_POOL_HAS_LIMIT:
                    common_pool_apples = min(real_common_pool, 2)
                else:
                    common_pool_apples = 3

            # Apple states, we know which ones they are in tinyMap, so we look for each of them if there is an apple
            #apple_state_4 = int(state[6] == 64)

            if forced_grass[0]:
                apple_state_1 = 1
            else:
                apple_state_1 = int(state[1] == 64)

            if forced_grass[1]:
                apple_state_2 = 1
            else:
                apple_state_2 = int(state[2] == 64)

            if forced_grass[2]:
                apple_state_3 = 1
            else:
                apple_state_3 = int(state[5] == 64)

            where_apples = apple_state_1 + 2*(apple_state_2 + 2*(apple_state_3))# + 2*apple_state_4)) # we encode them as a scalar, 8 different values

            # Total number of states: 16x6x8x6 = 4608 states (+1 if we count the state of being ill)
            position_and_apples = position + n_agent_cells*(agent_apples + n_apples*(where_apples + apples_in_ground*common_pool_apples))

            return int(position_and_apples)
    else:
        return state


def check_agents_where_apples(state):

    apple_pos_1 = int(state[1] > 64)
    apple_pos_2 = int(state[2] > 64)
    apple_pos_3 = int(state[5] > 64)

    return apple_pos_1, apple_pos_2, apple_pos_3



def check_redundant_states(positions_with_apples, agents_positions, ap_state):

    for n in range(len(positions_with_apples)):
        for ag_pos in agents_positions:
            if ag_pos == positions_with_apples[n]: #comprobamos que la posición del agente es una posición donde crecen manzanas
                if ap_state[n]: #comprobamos que esa posición "n" actualmente tiene una manzana
                    return True

    return False


def check_apples_state(state):
    apple_state_1 = state[1] == 64
    apple_state_2 = state[2] == 64
    apple_state_3 = state[5] == 64

    return apple_state_1, apple_state_2, apple_state_3


def check_agent_apples_state(agent, state):
    # Obtain the agent's real amount of apples
    return state[1 - 4 * number_of_agents + 4 * agent]


def check_common_pool(state):
    return state[-1]


def check_random_reward(state, actions):

    everyone_took_donation = True
    single_apple_in_pool = False

    if check_common_pool(state) == 1:
        single_apple_in_pool = True

    for action in actions:
        if action == 9:
            everyone_took_donation *= True
        else:
            everyone_took_donation *= False

    if everyone_took_donation and single_apple_in_pool:
        return True
    else:
        return False

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


def evaluation(env, tabularRL, policies=0):

    initial_state = env.reset()

    states = list()
    for ag in range(number_of_agents):
        states.append(new_state(ag, initial_state[ag], tabularRL))

    if policies == 0:
        policies = list()
        policies.append(policy_0)
        policies.append(policy_1)

    for t in range(2000):

        env.render()



        #actions = [policy_NULL[states[0]] for i in range(number_of_agents)]
        #actions[agent] = policy[states[agent]]


        actions = [policies[0][states[0]], policies[1][states[1]]]


        nObservations, rewards, nDone, _ = env.step(actions)

        print(states, actions, nObservations)
        print(rewards)

        states = list()
        for ag in range(number_of_agents):
            states.append(new_state(ag, nObservations[ag], tabularRL))

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




