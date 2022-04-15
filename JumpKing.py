#!/usr/env/bin python
#   
# Game Screen
# 

import pygame
import pygame.surfarray
import sys
import os
import uuid
import random
import copy
from collections import deque
import torch
import torch.nn as nn
from JumpKingGame import JKGame

import numpy as np
import time, datetime
import matplotlib.pyplot as plt

class JumpAgent:

    def __init__(self, state_dim, action_dim, save_dir):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir

        self.net = JumpNet(self.state_dim, self.action_dim).float()

        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            print("CUDA ENABLED <3")
            self.net = self.net.to(device="cuda")

        self.epsilon = 1
        self.epsilon_decay = 0.99999975
        self.epsilon_min = 0.1
        self.curr_step = 0

        self.save_every = 5e5  # no. of exp between saving mario net

        self.memory = deque(maxlen=100000)
        self.batch_size = 32

        self.gamma = 0.9

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()

        self.burnin = 1e4  # min. experiences before training
        self.learn_every = 3  # no. of experiences between updates to Q_online
        self.sync_every = 1e4  # no. of experiences between Q_target & Q_online sync

    def save(self):
        save_path = (
                self.save_dir / f"mario_net_{int(self.curr_step // self.save_every)}.chkpt"
        )
        torch.save(
            dict(model=self.net.state_dict(), epsilon=self.epsilon),
            save_path,
        )
        print(f"MarioNet saved to {save_path} at step {self.curr_step}")

    def sync_Q_With_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def td_estimate(self, state, action):
        current_Q = self.net(state, model="online")[
            np.arange(0, self.batch_size), action
        ]
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model="target")[
            np.arange(0, self.batch_size), best_action
        ]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    def cache(self, state, next_state, action, reward, done):
        if not torch.is_tensor(state):
            return
        state = state.__array__()
        next_state = next_state.__array__()

        if self.use_cuda:
            state = torch.tensor(state).cuda()
            next_state = torch.tensor(next_state).cuda()
            action = torch.tensor([action]).cuda()
            reward = torch.tensor([reward]).cuda()
            done = torch.tensor([done]).cuda()
        else:
            state = torch.tensor(state)
            next_state = torch.tensor(next_state)
            action = torch.tensor([action])
            reward = torch.tensor([reward])
            done = torch.tensor([done])
        # item will be automatically removed from left when maxLen is reached (circular buffer mechanism)
        self.memory.append((state, next_state, action, reward, done,))

    def recall(self):
        """Sample experiences from memory"""
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze() #squeeze just removes
    #redundant 1 dimensional axis's

    def act(self, state):
        """
    Given a state, choose an epsilon-greedy action and update value of step.

    Inputs:
    state(LazyFrame): A single observation of the current state, dimension is (state_dim)
    Outputs:
    action_idx (int): An integer representing which action Mario will perform
    """

        if np.random.rand() < self.epsilon:
            action_idx = np.random.randint(self.action_dim)
        else:
            # EXPLOIT
            state = state.__array__()
            if self.use_cuda:
                state = torch.tensor(state).cuda()
            else:
                state = torch.tensor(state)
            state = state.unsqueeze(0)
            action_values = self.net(state, model="online")
            action_idx = torch.argmax(action_values, axis=1).item()

        # decrease exploration rate
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)

        self.curr_step += 1
        return action_idx

    def train(self):
        """Update online action value (Q) function with a batch of experiences"""
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_With_target()

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None

        #only train learn_every
        if self.curr_step % self.learn_every != 0:
            return None, None

        # Sample from memory
        state, next_state, action, reward, done = self.recall()

        # Get TD Estimate
        td_est = self.td_estimate(state, action)

        # Get TD Target
        td_tgt = self.td_target(reward, next_state, done)

        # Backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)


class JumpNet(nn.Module):
    """mini cnn structure
  input -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output
  """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim

        if h != 112:
            raise ValueError(f"Expecting input height: 112, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: 84, got: {w}")

        self.online = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4480, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

        self.target = copy.deepcopy(self.online)

        # Q_target parameters are frozen.
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)


#IMPLEMENT BELOW WITH JUMPKINGGAME.PY
# Because consecutive frames don’t vary much, we can skip n-intermediate frames without losing much information.
# The n-th frame aggregates rewards accumulated over each skipped frame.
import argparse
import cv2
import os
# Converts 3d array to smaller 2d array in grayScale 84X84
from torchvision import transforms as T

class FrameStack:
    def __init__(self, image_array):
        self.image_array = image_array
        self.shape = 84

    def permute_orientation(self, observation):
        # permute [H, W, C] array to [C, H, W] tensor
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def resize_img(self, observation):
        transforms = T.Compose(
            [T.Resize(self.shape), T.Normalize(0, 255)]
        )
        observation = transforms(observation)

        return observation

    def convert_and_stack(self):
        final = []
        for image in self.image_array:
            observation = self.permute_orientation(image)
            transform = T.Grayscale()
            observation = transform(observation)

            observation = self.resize_img(observation)
            # plt.imshow(observation.permute(2, 1, 0))
            # plt.show()
            # time.sleep(2)
            final.append(observation)
        output = torch.stack((final[0], final[1], final[2], final[3]), dim=1)
        output = torch.squeeze(output)
        return output

def convert_screens_to_state(screens):
    return FrameStack(screens).convert_and_stack()



from pathlib import Path
from JumpKingMetricLogger import MetricLogger
def train():
    action_dict = {
        0: 'right',
        1: 'left',
        2: 'right+space',
        3: 'left+space',
        # 4: 'idle',
        # 5: 'space',
    }

    use_cuda = torch.cuda.is_available()
    print(f"Using CUDA: {use_cuda}")
    print()

    save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir.mkdir(parents=True)

    env = JKGame(max_step=1000)
    agent = JumpAgent(state_dim=(4, 112, 84), action_dim=4, save_dir=save_dir)
    logger = MetricLogger(save_dir)

    num_episode = 10000
    # model_load_path = 'agent_models/agent_model800'
    # agent.net.load_state_dict(torch.load(model_load_path))

    for e in range(num_episode):
        done, screen_arr = env.reset()

        #initial state 4 screens of start position
        state = convert_screens_to_state(screen_arr)
        agent_dict = agent.net.state_dict()
        running_reward = 0
        while not done:
            #run agent on state
            action = agent.act(state)

            #next state is returned as a list of ndarrays
            next_state, reward, done = env.step(action, agent_dict)

            new_stack = FrameStack(next_state).convert_and_stack()

            # Remember
            agent.cache(state, new_stack, action, reward, done)

            running_reward += reward

            #learn
            q, loss = agent.train()

            # Logging
            logger.log_step(reward, loss, q)

            #update state
            state = new_stack
        print(f'episode: {e}, reward: {running_reward}')
        logger.log_episode()
        if e % 20 == 0:
            logger.record(episode=e, epsilon=agent.epsilon, step=agent.curr_step)



if __name__ == "__main__":
    # Game = JKGame()
    # Game.running()
    train()
