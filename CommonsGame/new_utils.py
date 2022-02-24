from constants import TOO_MANY_APPLES

number_of_agents = 2
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
common_pool_states = range(3)

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


def new_state(agent, state, tabularRL, forced_agent_apples=-1, forced_grass=[False, False, False], forced_ag_pos=[-1, -1]):
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


            if forced_ag_pos[0] > -1 and forced_ag_pos[1] > -1:
                agent_x = forced_ag_pos[0]
                agent_y = forced_ag_pos[1]
            else:
                agent_x = state[-1 - 4 * number_of_agents + 4 * agent]
                agent_y = state[-4 * number_of_agents + 4 * agent]

            position = agent_x + agent_x_position*agent_y  # we encode them as a scalar, there are 16 different positions

            # Obtain the agent's amount of apples, but we only consider four possible values
            if forced_agent_apples > -1:
                agent_apples = forced_agent_apples
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

            common_pool_apples = min(state[-1], common_pool_max - 1) # 6 different values

            # Apple states, we know which ones they are in tinyMap, so we look for each of them if there is an apple:



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



