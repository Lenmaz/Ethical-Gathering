import random

import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import spaces
from pycolab import ascii_art
from CommonsGame.utils import buildMap, ObservationToArrayWithRGB
from CommonsGame.objects import PlayerSprite, AppleDrape, SightDrape, ShotDrape
from CommonsGame.constants import TIMEOUT_FRAMES, SUSTAINABILITY_MATTERS


class CommonsGame(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

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

    def __init__(self, numAgents, visualRadius, mapSketch, fullState, tabularState, agent_pos=[]):
        super(CommonsGame, self).__init__()
        self.fullState = fullState
        # Setup spaces
        self.action_space = spaces.Discrete(10)
        obHeight = obWidth = visualRadius * 2 + 1
        # Setup game
        self.numAgents = numAgents
        self.sightRadius = visualRadius
        self.agentChars = agentChars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"[0:numAgents]
        self.mapHeight = len(mapSketch)
        self.mapWidth = len(mapSketch[0])
        self.tabularState = tabularState
        self.common_pool = True
        self.agent_pos = agent_pos

        if tabularState:
            fullState = True
            self.fullState = True

        if fullState:
            self.observation_space = spaces.Box(low=0, high=255, shape=(self.mapHeight + 2, self.mapWidth + 2, 3),
                                                dtype=np.uint8)
        else:
            self.observation_space = spaces.Box(low=0, high=255, shape=(obHeight, obWidth, 3), dtype=np.uint8)
        self.numPadPixels = numPadPixels = visualRadius - 1

        self.gameField = buildMap(mapSketch, numPadPixels=numPadPixels, agentChars=agentChars, mandatory_initial_position=self.agent_pos)

        self.state = None
        self.sick_probabilities = np.random.choice(100, numAgents)
        self.efficiency_probabilites = np.random.randint(1, 5, numAgents)

        # Pycolab related setup:
        self._game = self.buildGame()
        colourMap = dict([(a, (999-4*i, 0, 4*i)) for i, a in enumerate(agentChars)]  # Agents
                         + [('=', (705, 705, 705))]  # Steel Impassable wall
                         + [(' ', (0, 0, 0))]  # Black background
                         + [('@', (0, 999, 0))]  # Green Apples
                         + [('.', (750, 750, 0))]  # Yellow beam
                         + [('-', (0, 0, 0))])  # Grey scope
        self.obToImage = ObservationToArrayWithRGB(colour_mapping=colourMap)

    def buildGame(self, apples_yes_or_not=[True, True, True]):
        agentsOrder = list(self.agentChars)
        random.shuffle(agentsOrder)

        return ascii_art.ascii_art_to_game(
            self.gameField,
            what_lies_beneath=' ',
            sprites=dict(
                [(a, ascii_art.Partial(PlayerSprite, self.agentChars, self.sightRadius, self.agent_pos)) for a in self.agentChars]),
            drapes={'@': ascii_art.Partial(AppleDrape, self.agentChars, self.numPadPixels, apples_yes_or_not),
                    '-': ascii_art.Partial(SightDrape, self.agentChars, self.numPadPixels),
                    '.': ascii_art.Partial(ShotDrape, self.agentChars, self.numPadPixels)},
            # update_schedule=['.'] + agentsOrder + ['-'] + ['@'],
            update_schedule=['.'] + agentsOrder + ['-'] + ['@'],
            z_order=['-'] + ['@'] + agentsOrder + ['.']
        )

    def step(self, nActions):
        nInfo = {'n': []}
        self.state, nRewards, _ = self._game.play(nActions)


        nObservations, done = self.getObservation()
        nDone = [done] * self.numAgents
        return nObservations, nRewards, nDone, nInfo

    def reset(self, num_apples=[0], common_pool=0, apples_yes_or_not=[True, True, True]):
        # Reset the state of the environment to an initial state
        self._game = self.buildGame(apples_yes_or_not)
        ags = [self._game.things[c] for c in self.agentChars]



        for i, a in enumerate(ags):
            a.set_sickness(self.sick_probabilities[i])
            #a.set_efficiency(self.efficiency_probabilites[i]) #TODO: Not random efficiency
            if len(num_apples) == 1:
                a.set_init_apples(num_apples[0]) # all agents have the same amount of apples, why not?
            else:
                a.set_init_apples(num_apples[i])
        self._game.things['@'].common_pool = common_pool

        self.state, _, _ = self._game.its_showtime()
        nObservations, _ = self.getObservation()
        return nObservations

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        board = self.obToImage(self.state)['RGB'].transpose([1, 2, 0])
        board = board[self.numPadPixels:self.numPadPixels + self.mapHeight + 2,
                self.numPadPixels:self.numPadPixels + self.mapWidth + 2, :]
        plt.figure(1)
        plt.imshow(board)
        plt.axis("off")

        ags = [self._game.things[c] for c in self.agentChars]
        plot_text = ""
        for i, agent in enumerate(ags):
            plot_text += "Agent "+ str(i+1) + ": " + str(agent.has_apples) + ", "
        plot_text += "Common: " + str(self._game.things['@'].common_pool)
        plt.title(plot_text)
        plt.show(block=False)
        # plt.show()
        plt.pause(0.05)
        plt.clf()

    def getObservation(self):
        if SUSTAINABILITY_MATTERS:
            done = not (np.logical_or.reduce(self.state.layers['@'][self.sightRadius + 2:,:], axis=None))
        else:
            done = False
        ags = [self._game.things[c] for c in self.agentChars]
        obs = []

        new_state = self.state.board[self.sightRadius + 2:-self.sightRadius, self.sightRadius:-self.sightRadius]
        common_apples = self._game.things['@'].common_pool
        board = self.obToImage(self.state)['RGB'].transpose([1, 2, 0])
        num_agents = 0
        for a in ags:
            if a.visible or a.timeout == TIMEOUT_FRAMES:
                if self.fullState:

                    ob = np.copy(board)
                    if a.visible:
                        ob[a.position[0], a.position[1], :] = [0, 0, 255]
                    ob = ob[self.numPadPixels:self.numPadPixels + self.mapHeight + 2,
                         self.numPadPixels:self.numPadPixels + self.mapWidth + 2, :]

                else:
                    ob_apples = np.copy(board[4, 4 + 3*num_agents:5 + 1 + 3*num_agents, :])
                    relleno = np.copy(board[0, : 2*self.sightRadius - 1, :])

                    ob = np.copy(board[
                                 a.position[0] - self.sightRadius:a.position[0] + self.sightRadius + 1,
                                 a.position[1] - self.sightRadius:a.position[1] + self.sightRadius + 1, :])

                    ob_apples = np.vstack((ob_apples, relleno))
                    ob = np.concatenate((ob, np.array([ob_apples])))
                    #print(ob)
                    if a.visible:
                        ob[self.sightRadius, self.sightRadius, :] = [0, 0, 255]
                ob = ob / 255.0
            else:
                ob = None
            new_state = np.append(new_state, [a.position[0] - self.sightRadius - 2, a.position[1] - self.sightRadius, a.has_apples, a.donated_apples])

            if not self.tabularState:
                obs.append(ob)
            num_agents += 1
        new_state = np.append(new_state, [common_apples])
        #print("State : ", new_state)

        if self.tabularState:
            for a in ags:
                if a.visible or a.timeout == TIMEOUT_FRAMES:
                    obs.append(new_state)
                else:
                    obs.append([])
        return obs, done

    def get_agents(self):
        return [self._game.things[c] for c in self.agentChars]
