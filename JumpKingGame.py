#!/usr/env/bin python
#
# Game Screen
#

import pygame
import sys
import os
import uuid

import inspect
import pickle
import numpy as np
from environment import Environment
from spritesheet import SpriteSheet
from Background import Backgrounds
from King import King
from Babe import Babe
from Level import Levels
from Menu import Menus
import torch
from Start import Start
from collections import deque
import itertools


class JKGame:
    """ Overall class to manga game aspects """

    def __init__(self, max_step=float('inf'),):

        pygame.init()

        self.environment = Environment()

        self.clock = pygame.time.Clock()

        self.fps = 800

        self.bg_color = (0, 0, 0)

        self.screen = pygame.display.set_mode(
                                              (int(os.environ.get("screen_width")) * int(os.environ.get("window_scale")), int(os.environ.get("screen_height")) * int
                                                  (os.environ.get("window_scale"))), pygame.HWSURFACE |pygame.DOUBLEBUF  )  # |pygame.SRCALPHA)

        self.game_screen = pygame.Surface((int(os.environ.get("screen_width")), int(os.environ.get("screen_height"))), pygame.HWSURFACE |pygame.DOUBLEBUF  )  # |pygame.SRCALPHA)

        self.game_screen_x = 0

        pygame.display.set_icon(pygame.image.load("images\\sheets\\JumpKingIcon.ico"))

        self.levels = Levels(self.game_screen)

        self.king = King(self.game_screen, self.levels)

        self.babe = Babe(self.game_screen, self.levels)

        self.menus = Menus(self.game_screen, self.levels, self.king)

        self.start = Start(self.game_screen, self.menus)

        self.step_counter = 0
        self.max_step = max_step

        self.saved_frames = deque(maxlen=10000) #save every non-skipped frame (n=4)
        self.frame_counter = 0
        self.frame_skip = 4
        self.visited = {}
        self.levels_visited = {}

        pygame.display.set_caption('Jump King At Home XD TriHard')

    def reset(self):
        self.king.reset()
        self.levels.reset()
        os.environ["start"] = "1"
        os.environ["gaming"] = "1"
        os.environ["pause"] = ""
        os.environ["active"] = "1"
        os.environ["attempt"] = str(int(os.environ.get("attempt")) + 1)
        os.environ["session"] = "0"

        self.step_counter = 0
        done = False
        screen = pygame.surfarray.array3d(pygame.display.get_surface())

        self.visited = {}
        self.visited[(self.king.levels.current_level, self.king.y)] = 1

        return done, [screen, screen, screen, screen]

    def move_available(self):
        available = not self.king.isFalling \
                    and not self.king.levels.ending \
                    and (not self.king.isSplat or self.king.splatCount > self.king.splatDuration)
        return available


    def step(self, action, model_dict=None):
        # isnt this an issue that the old_level, old_y value is always the spawn position?
        old_level = self.king.levels.current_level
        old_y = self.king.y
        # old_y = (self.king.levels.max_level - self.king.levels.current_level) * 360 + self.king.y
        reward = 0
        # This while true is needed to wait for the agent if he is mid jump to come down
        while True:
            self.clock.tick() #self.fps once per frame, this returns how many ms have passed since previous call

            if self.frame_counter % self.frame_skip == 0:  # save every 4th frame
                self.saved_frames.append(pygame.surfarray.array3d(pygame.display.get_surface()))
            self.frame_counter += 1

            self._check_events(model_dict)
            if not os.environ["pause"]:
                if not self.move_available():
                    action = None
                self._update_gamestuff(action=action)

            self._update_gamescreen()
            # self._update_guistuff()
            # self._update_audio()
            # pygame.display.update()

            if self.move_available():
                self.step_counter += 1
                # state = [self.king.levels.current_level, self.king.x, self.king.y, self.king.jumpCount]
                ##################################################################################################
                # Define the reward from environment                                                             #
                ##################################################################################################
                # if we got to a new level, or if we have a higher y position, give us a positive  reward!
                if self.king.levels.current_level > old_level and self.king.levels.current_level not in self.levels_visited:
                    reward += 50000 * self.king.levels.current_level
                elif self.king.levels.current_level < old_level:
                    print("OH NO WENT BACK A LVL")
                    reward += -10000
                elif self.king.levels.current_level == old_level and self.king.y < old_y:
                    reward += (self.king.levels.current_level + 1) * 10  # used to be just 0, now we give reward for being in a higher level
                else:
                    # negative reward + 1
                    self.visited[(self.king.levels.current_level, self.king.y)] = self.visited.get((self.king.levels.current_level, self.king.y), 0) + 1
                    # if our reward and y visited value is less than the episode starting position reward and y visited value, set it to the old +1
                    if (old_level, old_y) in self.visited and self.visited[(self.king.levels.current_level, self.king.y)] < self.visited[(old_level, old_y)]:
                        self.visited[(self.king.levels.current_level, self.king.y)] = self.visited[(old_level, old_y)] + 1

                    reward += -self.visited[(self.king.levels.current_level, self.king.y)]
                ####################################################################################################
                self.levels_visited[self.king.levels.current_level] = True
                done = True if self.step_counter > self.max_step else False
                #return state
                if len(self.saved_frames) >= self.frame_skip:
                    state = list(itertools.islice(self.saved_frames, self.frame_skip))
                    self.saved_frames.popleft()
                    return state, reward, done


    def running(self):
        """
        play game with keyboard
        :return:
        """
        self.reset()
        while True:
            # state = [self.king.levels.current_level, self.king.x, self.king.y, self.king.jumpCount]
            # print(state)
            self.clock.tick(self.fps)
            self._check_events()
            if not os.environ["pause"]:
                self._update_gamestuff()

            self._update_gamescreen()
            self._update_guistuff()
            self._update_audio()
            pygame.display.update()

    def _check_events(self, model_dict=None):

        for event in pygame.event.get():

            if event.type == pygame.QUIT:

                self.environment.save()

                self.menus.save()

                sys.exit()

            if event.type == pygame.KEYDOWN:

                self.menus.check_events(event)
                if event.key == pygame.K_ESCAPE:
                    if model_dict is not None:
                        torch.save(model_dict, 'agent_models/agent_model' + str(uuid.uuid4()))
                    sys.exit()
                if event.key == pygame.K_c:

                    if os.environ["mode"] == "creative":

                        os.environ["mode"] = "normal"

                    else:

                        os.environ["mode"] = "creative"

            if event.type == pygame.VIDEORESIZE:

                self._resize_screen(event.w, event.h)

    def _update_gamestuff(self, action=None):

        self.levels.update_levels(self.king, self.babe, agentCommand=action)

    def _update_guistuff(self):

        if self.menus.current_menu:

            self.menus.update()

        if not os.environ["gaming"]:

            self.start.update()

    def _update_gamescreen(self):

        pygame.display.set_caption(f"Jump King At Home XD - {self.clock.get_fps():.2f} FPS")

        self.game_screen.fill(self.bg_color)

        if os.environ["gaming"]:

            self.levels.blit1()

        if os.environ["active"]:

            self.king.blitme()

        if os.environ["gaming"]:

            self.babe.blitme()

        if os.environ["gaming"]:

            self.levels.blit2()

        if os.environ["gaming"]:

            self._shake_screen()

        if not os.environ["gaming"]:

            self.start.blitme()

        self.menus.blitme()

        self.screen.blit(pygame.transform.scale(self.game_screen, self.screen.get_size()), (self.game_screen_x, 0))

    def _resize_screen(self, w, h):

        self.screen = pygame.display.set_mode((w, h), pygame.HWSURFACE |pygame.DOUBLEBUF |pygame.SRCALPHA)

    def _shake_screen(self):

        try:

            if self.levels.levels[self.levels.current_level].shake:

                if self.levels.shake_var <= 150:

                    self.game_screen_x = 0

                elif self.levels.shake_var // 8 % 2 == 1:

                    self.game_screen_x = -1

                elif self.levels.shake_var // 8 % 2 == 0:

                    self.game_screen_x = 1

            if self.levels.shake_var > 260:

                self.levels.shake_var = 0

            self.levels.shake_var += 1

        except Exception as e:

            print("SHAKE ERROR: ", e)

    def _update_audio(self):

        for channel in range(pygame.mixer.get_num_channels()):

            if not os.environ["music"]:

                if channel in range(0, 2):

                    pygame.mixer.Channel(channel).set_volume(0)

                    continue

            if not os.environ["ambience"]:

                if channel in range(2, 7):

                    pygame.mixer.Channel(channel).set_volume(0)

                    continue

            if not os.environ["sfx"]:

                if channel in range(7, 16):

                    pygame.mixer.Channel(channel).set_volume(0)

                    continue

            pygame.mixer.Channel(channel).set_volume(float(os.environ.get("volume")))