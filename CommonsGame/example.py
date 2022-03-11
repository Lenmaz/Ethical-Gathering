import numpy as np
import gym
from constants import tinyMap, smallMap, bigMap

numAgents = 2

env = gym.make('CommonsGame:CommonsGame-v0', numAgents=numAgents, visualRadius=3, mapSketch=tinyMap,
               fullState=False, tabularState=True, agent_pos=[[4, 1], [4, 1]])
initial_state = env.reset(num_apples=1, common_pool=0)

print(env.observation_space)

MOVE_UP = 0
MOVE_DOWN = 1
MOVE_LEFT = 2
MOVE_RIGHT = 3

TURN_CLOCKWISE = 4
TURN_COUNTERCLOCKWISE = 5
STAY = 6
SHOOT = 7
DONATE = 8
TAKE_DONATION = 9


print(initial_state[0])

for t in range(100):

    #nActions = np.random.randint(low=0, high=env.action_space.n, size=(numAgents,)).tolist()
    nActions = [DONATE, TAKE_DONATION]
    nObservations, nRewards, nDone, nInfo = env.step(nActions)

    print(nObservations[0], nRewards)
    print("--Time step", t ,"--")
    env.render()

common_apples = 0
for n, agent in enumerate(env.get_agents()):
    print("Agent")
    print("Agent " + str(n) + " possessions : " + str(agent.has_apples))
    print("Agent " + str(n) + " donations : " + str(agent.donated_apples))
    print("Agent " + str(n) + "'s efficiency : " + str(agent.efficiency))
    common_apples += agent.donated_apples

    print("--")

print("Total common apples : ", common_apples)
