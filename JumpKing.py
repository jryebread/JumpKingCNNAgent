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

from Start import Start

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from JumpKingGame import JKGame
import torch.nn.functional as F
import torch.optim as optim
import random
import time





def train():
	action_dict = {
		0: 'right',
		1: 'left',
		2: 'right+space',
		3: 'left+space',
		# 4: 'idle',
		# 5: 'space',
	}
	agent = DDQN()
	env = JKGame(max_step=1000)
	num_episode = 10000
	model_load_path = 'agent_models/agent_model800'
	agent.eval_net.load_state_dict(torch.load(model_load_path))
	rewards = []
	for i in range(num_episode):
		done, state = env.reset()
		agent_dict = agent.eval_net.state_dict()
		running_reward = 0
		while not done:
			action = agent.select_action(state)
			#print(action_dict[action])
			# if i % 10 == 0:
			# 	print(agent.epsilon)

			next_state, reward, done = env.step(action, agent_dict)

			running_reward += reward
			rewards.append(reward)
			sign = 1 if done else 0
			agent.train(state, action, reward, next_state, sign)
			state = next_state
		print (f'episode: {i}, reward: {running_reward}')


	torch.save(agent.eval_net.state_dict(), 'agent_models/agent_model' + str(uuid.uuid4()))
	plt.plot(rewards)
	plt.savefig('myfig')


if __name__ == "__main__":
	# Game = JKGame()
	# Game.running()
	train()
