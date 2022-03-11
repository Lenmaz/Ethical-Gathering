import numpy as np
from pycolab.prefab_parts import sprites
from pycolab import things as pythings
from scipy.ndimage import convolve
from CommonsGame.constants import *



class PlayerSprite(sprites.MazeWalker):
    def __init__(self, corner, position, character, agentChars, forced_pos=[]):
        super(PlayerSprite, self).__init__(
            corner, position, character, impassable=['='] #+ list(agentChars.replace(character, ''))
            ,confined_to_board=True)
        self.agentChars = agentChars
        self.orientation = 0

        if len(forced_pos) > 0:
            if isinstance(forced_pos[0], int):
                this_forced_pos = forced_pos
            else:
                this_forced_pos = forced_pos[ord(character) - 65]
            self.initPos = this_forced_pos
            self._teleport((int(this_forced_pos[0] + 3), int(this_forced_pos[1] + 3)))  #TODO: +3 RELATED WITH SIGHTRADIUS
        else:
            self.initPos = position

        self.visualRadius = 0
        self.timeout = 0

        # New variables
        self.has_apples = 0
        self.has_shot = False
        self.has_donated = False
        self.is_sick = False
        self.did_nothing = False
        self.took_donation = False

        self.efficiency = 1 if character == "A" else 4

        self.probability_getting_sick = 0
        self.donated_apples = 0

    def set_init_apples(self, num_apples):
        self.has_apples = num_apples

    def set_sickness(self, prob):
        if 0 <= prob <= 100:
            if AGENTS_CAN_GET_SICK:
                self.probability_getting_sick = prob

    def set_efficiency(self, prob):
        if 1 <= prob <= 6:
            if AGENTS_HAVE_DIFFERENT_EFFICIENCY:
                self.efficiency = prob

    def update(self, actions, board, layers, backdrop, things, the_plot):

        self.is_sick = self.probability_getting_sick > np.random.choice(100)
        if actions is not None:
            a = actions[self.agentChars.index(self.character)]
        else:
            return
        if self._visible:
            if things['.'].curtain[self.position[0], self.position[1]] or self.is_sick:
                self.timeout = TIMEOUT_FRAMES
                self._visible = False
            else:
                if a == 0:  # go upward?
                    if self.orientation == 0:
                        self._north(board, the_plot)
                    elif self.orientation == 1:
                        self._east(board, the_plot)
                    elif self.orientation == 2:
                        self._south(board, the_plot)
                    elif self.orientation == 3:
                        self._west(board, the_plot)
                elif a == 1:  # go downward?
                    if self.orientation == 0:
                        self._south(board, the_plot)
                    elif self.orientation == 1:
                        self._west(board, the_plot)
                    elif self.orientation == 2:
                        self._north(board, the_plot)
                    elif self.orientation == 3:
                        self._east(board, the_plot)
                elif a == 2:  # go leftward?
                    if self.orientation == 0:
                        self._west(board, the_plot)
                    elif self.orientation == 1:
                        self._north(board, the_plot)
                    elif self.orientation == 2:
                        self._east(board, the_plot)
                    elif self.orientation == 3:
                        self._south(board, the_plot)
                elif a == 3:  # go rightward?
                    if self.orientation == 0:
                        self._east(board, the_plot)
                    elif self.orientation == 1:
                        self._south(board, the_plot)
                    elif self.orientation == 2:
                        self._west(board, the_plot)
                    elif self.orientation == 3:
                        self._north(board, the_plot)
                elif a == 4:  # turn right?
                    if self.orientation == 3:
                        self.orientation = 0
                    else:
                        self.orientation = self.orientation + 1
                elif a == 5:  # turn left?
                    if self.orientation == 0:
                        self.orientation = 3
                    else:
                        self.orientation = self.orientation - 1
                elif a == 6:  # do nothing?
                    self.did_nothing = True
                    self._stay(board, the_plot)
                elif a == 8:  # donate?
                    if self.has_apples > 0:
                        self.has_donated = True
                    self._stay(board, the_plot)
                elif a == 9:  # took donation?
                    self.took_donation = True
                    self._stay(board, the_plot)
        else:
            if self.timeout == 0:
                self._teleport(self.initPos)
                self._visible = True
            else:
                self.timeout -= 1


class SightDrape(pythings.Drape):
    """Scope of agent Drap"""
    def __init__(self, curtain, character, agentChars, numPadPixels):
        super().__init__(curtain, character)
        self.agentChars = agentChars
        self.numPadPixels = numPadPixels
        self.h = curtain.shape[0] - (numPadPixels * 2 + 2)
        self.w = curtain.shape[1] - (numPadPixels * 2 + 2)

    def update(self, actions, board, layers, backdrop, things, the_plot):
        np.logical_and(self.curtain, False, self.curtain)
        ags = [things[c] for c in self.agentChars]
        for agent in ags:
            if agent.visible:
                pos = agent.position
                if agent.orientation == 0:
                    self.curtain[pos[0] - 1, pos[1]] = True
                elif agent.orientation == 1:
                    self.curtain[pos[0], pos[1] + 1] = True
                elif agent.orientation == 2:
                    self.curtain[pos[0] + 1, pos[1]] = True
                elif agent.orientation == 3:
                    self.curtain[pos[0], pos[1] - 1] = True
                self.curtain[:, :] = np.logical_and(self.curtain, np.logical_not(layers['=']))


class ShotDrape(pythings.Drape):
    """Tagging ray Drap"""
    def __init__(self, curtain, character, agentChars, numPadPixels):
        super().__init__(curtain, character)
        self.agentChars = agentChars
        self.numPadPixels = numPadPixels
        self.h = curtain.shape[0] - (numPadPixels * 2 + 2)
        self.w = curtain.shape[1] - (numPadPixels * 2 + 2)
        self.scopeHeight = numPadPixels + 1

    def update(self, actions, board, layers, backdrop, things, the_plot):
        beamWidth = 0
        beamHeight = self.scopeHeight
        np.logical_and(self.curtain, False, self.curtain)
        if actions is not None:
            for i, a in enumerate(actions):
                if a == 7:
                    things[self.agentChars[i]].has_shot = True
                    agent = things[self.agentChars[i]]
                    if agent.visible:
                        pos = agent.position
                        if agent.orientation == 0:
                            if np.any(layers['='][pos[0] - beamHeight:pos[0],
                                      pos[1] - beamWidth:pos[1] + beamWidth + 1]):
                                collisionIdxs = np.argwhere(layers['='][pos[0] - beamHeight:pos[0],
                                                            pos[1] - beamWidth:pos[1] + beamWidth + 1])
                                beamHeight = beamHeight - (np.max(collisionIdxs) + 1)
                            self.curtain[pos[0] - beamHeight:pos[0],
                            pos[1] - beamWidth:pos[1] + beamWidth + 1] = True
                        elif agent.orientation == 1:
                            if np.any(layers['='][pos[0] - beamWidth:pos[0] + beamWidth + 1,
                            pos[1] + 1:pos[1] + beamHeight + 1]):
                                collisionIdxs = np.argwhere(layers['='][pos[0] - beamWidth:pos[0] + beamWidth + 1,
                            pos[1] + 1:pos[1] + beamHeight + 1])
                                beamHeight = np.min(collisionIdxs)
                            self.curtain[pos[0] - beamWidth:pos[0] + beamWidth + 1,
                            pos[1] + 1:pos[1] + beamHeight + 1] = True
                        elif agent.orientation == 2:
                            if np.any(layers['='][pos[0] + 1:pos[0] + beamHeight + 1,
                            pos[1] - beamWidth:pos[1] + beamWidth + 1]):
                                collisionIdxs = np.argwhere(layers['='][pos[0] + 1:pos[0] + beamHeight + 1,
                            pos[1] - beamWidth:pos[1] + beamWidth + 1])
                                beamHeight = np.min(collisionIdxs)
                            self.curtain[pos[0] + 1:pos[0] + beamHeight + 1,
                            pos[1] - beamWidth:pos[1] + beamWidth + 1] = True
                        elif agent.orientation == 3:
                            if np.any(layers['='][pos[0] - beamWidth:pos[0] + beamWidth + 1,
                                      pos[1] - beamHeight:pos[1]]):
                                collisionIdxs = np.argwhere(layers['='][pos[0] - beamWidth:pos[0] + beamWidth + 1,
                                                            pos[1] - beamHeight:pos[1]])
                                beamHeight = beamHeight - (np.max(collisionIdxs) + 1)
                            self.curtain[pos[0] - beamWidth:pos[0] + beamWidth + 1, pos[1] - beamHeight:pos[1]] = True
                        # self.curtain[:, :] = np.logical_and(self.curtain, np.logical_not(layers['=']))
        else:
            return


class AppleDrape(pythings.Drape):
    """Coins Drap"""
    def __init__(self, curtain, character, agentChars, numPadPixels, apples_yes_or_not):
        super().__init__(curtain, character)
        self.agentChars = agentChars
        self.numPadPixels = numPadPixels

        self.apples = np.copy(curtain)
        self.curtain[5, 4] = apples_yes_or_not[0]
        self.curtain[5, 5] = apples_yes_or_not[1]
        self.curtain[6, 4] = apples_yes_or_not[2]

        self.max_apples_in_ground = 3

        self.all_agents_deserve_apple = False

        self.common_pool = 0
        self.common_pool_had_one_apple = False
        self.both_agents_took_donation = False

    def update(self, actions, board, layers, backdrop, things, the_plot):
        rewards = []
        agents_together = False
        agentsMap = np.ones(self.curtain.shape, dtype=bool)

        pos_x = [-9, -9]
        pos_y = [-9, -9]
        for i in range(len(self.agentChars)):
            pos_x[i] = things[self.agentChars[i]].position[0]
            pos_y[i] = things[self.agentChars[i]].position[1]

        if pos_x[0] == pos_x[1] and pos_y[0] == pos_y[1]:
            agents_together = True

        if len(self.agentChars) > 1:
            self.both_agents_took_donation = True
            for i in range(len(self.agentChars)):
                self.both_agents_took_donation *= things[self.agentChars[i]].took_donation

        agent_efficiencies = []

        if len(self.agentChars) > 1:
            for i in range(len(self.agentChars)):
                agent_efficiencies.append(things[self.agentChars[i]].efficiency) # The number of apples it can collect on each turn

        cannot_receive_yet = False #only when the common pool had 0 apples, one agent donates and the other tries to take it in the same turn


        for i in range(len(self.agentChars)):

            agent_efficiency = things[self.agentChars[i]].efficiency # The number of apples it can collect on each turn

            rew = self.curtain[things[self.agentChars[i]].position[0], things[self.agentChars[i]].position[1]]

            greedy = False  # A greedy agent takes more apples than what it needs
            not_stupid = False # A stupid agent does not take more apples when it needs them
            hungry = False
            apple_lost = False

            if things[self.agentChars[i]].has_apples < TOO_MANY_APPLES:
                hungry = True

            if rew:
                if agents_together:
                    if agent_efficiency == max(agent_efficiencies):
                        self.curtain[
                        things[self.agentChars[i]].position[0], things[self.agentChars[i]].position[1]] = False
                        things[self.agentChars[i]].has_apples += 1
                    else:
                        pass
                else:
                    self.curtain[things[self.agentChars[i]].position[0], things[self.agentChars[i]].position[1]] = False
                    things[self.agentChars[i]].has_apples += 1


            #elif things[self.agentChars[i]].did_nothing:
            #    not_stupid = things[self.agentChars[i]].has_apples > TOO_MANY_APPLES
            #    things[self.agentChars[i]].did_nothing = False
            #else:
            #    things[self.agentChars[i]].did_nothing = False

            donation = things[self.agentChars[i]].has_donated
            took_donation = things[self.agentChars[i]].took_donation
            shot = things[self.agentChars[i]].has_shot
            if donation:

                if things[self.agentChars[i]].has_apples < TOO_MANY_APPLES + 1:

                    donation = False   ## If the agent itself doesn't have enough apples, it isn't a donation, it's more like an extortion

                    if things[self.agentChars[i]].has_apples == TOO_MANY_APPLES:

                        apple_lost = True

                if self.common_pool >= TOO_MANY_APPLES*len(self.agentChars):
                    donation = False ## No ethical reward if the common pool already has a lot of apples


                things[self.agentChars[i]].has_donated = False
                things[self.agentChars[i]].has_apples -= 1
                things[self.agentChars[i]].donated_apples += 1
                self.common_pool += 1


                if self.common_pool == 1:
                    cannot_receive_yet = True

            elif took_donation:
                things[self.agentChars[i]].took_donation = False

                apples_taken_in_ground = np.argwhere(np.logical_and(np.logical_xor(self.apples, self.curtain), agentsMap))

                #if len(apples_taken_in_ground) < self.max_apples_in_ground: #you can only take donations if there are no apples left in ground
                #if False:
                #    took_donation = False
                #else:
                if self.common_pool > 0 or self.common_pool_had_one_apple:
                    greedy = not hungry or len(apples_taken_in_ground) < self.max_apples_in_ground

                    if self.common_pool > 0:

                        if cannot_receive_yet:
                            took_donation = False
                        else:
                            self.common_pool -= 1

                        if self.common_pool == 0:
                            self.common_pool_had_one_apple = True

                    if self.common_pool == 0 and self.both_agents_took_donation:
                        if self.common_pool_had_one_apple:
                            if agent_efficiency == max(agent_efficiencies):
                                things[self.agentChars[i]].has_apples += 1
                                self.common_pool_had_one_apple = False
                            else:
                                took_donation = False
                        else:
                            took_donation = False

                    else:
                        if not cannot_receive_yet:
                            things[self.agentChars[i]].has_apples += 1
                            self.common_pool_had_one_apple = False

                else:
                    took_donation = False  # To guarantee that it only receives reward if there are apples
            elif shot:
                things[self.agentChars[i]].has_shot = False

            if things[self.agentChars[i]].timeout > 0:
                rewards.append(0)
            else:
                # The rewards takes into account if an apple has been gathered or if an apple has been donated
                individual_reward = rew*APPLE_GATHERING_REWARD + took_donation*TOOK_DONATION_REWARD + hungry*HUNGER
                ethical_reward = donation*DONATION_REWARD + shot*SHOOTING_PUNISHMENT + greedy*TOO_MANY_APPLES_PUNISHMENT + apple_lost*LOST_APPLE
                sustain_reward = + not_stupid*DID_NOTHING_BECAUSE_MANY_APPLES_REWARD

                # for Single_Objective you need: the_reward = individual_reward
                #the_reward = individual_reward

                the_reward = [individual_reward, ethical_reward]

                #print(the_reward)

                rewards.append(the_reward)

            agentsMap[things[self.agentChars[i]].position[0], things[self.agentChars[i]].position[1]] = False


        the_plot.add_reward(rewards)
        # Matrix of local stock of apples
        kernel = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 0, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
        L = convolve(self.curtain[self.numPadPixels + 1:-self.numPadPixels - 1,
                     self.numPadPixels + 1:-self.numPadPixels - 1] * 1, kernel, mode='constant')
        probs = np.zeros(L.shape)

        probs[(L > 0) & (L <= 2)] = respawnProbs[0]
        probs[(L > 2) & (L <= 4)] = respawnProbs[1]
        probs[(L > 4)] = respawnProbs[2]

        ags = [things[c] for c in self.agentChars]
        num_agent = 0

        x_agent = self.numPadPixels + 1

        y_agent = self.numPadPixels + 1

        for agent in ags:
            if agent.has_apples > 1:
                self.apples[x_agent, y_agent + 3*num_agent] = True
                self.curtain[x_agent, y_agent + 3*num_agent] = True

                self.apples[x_agent, y_agent + 1 + 3*num_agent] = True
                self.curtain[x_agent, y_agent + 1 + 3*num_agent] = True
            elif agent.has_apples > 0:
                self.apples[x_agent, y_agent + 3*num_agent] = True
                self.curtain[x_agent, y_agent + 3*num_agent] = True

                self.apples[x_agent, y_agent + 1 + 3*num_agent] = False
                self.curtain[x_agent, y_agent + 1 + 3*num_agent] = False
            else:
                self.apples[x_agent, y_agent + 3*num_agent] = False
                self.curtain[x_agent, y_agent + 3*num_agent] = False

                self.apples[x_agent, y_agent + 1 + 3*num_agent] = False
                self.curtain[x_agent, y_agent + 1 + 3*num_agent] = False
            num_agent += 1


        appleIdxs = np.argwhere(np.logical_and(np.logical_xor(self.apples, self.curtain), agentsMap))


        for i, j in appleIdxs:
            if SUSTAINABILITY_MATTERS:
                pass
                #self.curtain[i, j] = np.random.choice([True, False],
                #                                  p=[probs[i - self.numPadPixels - 1, j - self.numPadPixels - 1],
                #                                     1 - probs[i - self.numPadPixels - 1, j - self.numPadPixels - 1]])

            else:
                self.curtain[i, j] = np.random.choice([True, False], p=[REGENERATION_PROBABILITY, 1-REGENERATION_PROBABILITY])
