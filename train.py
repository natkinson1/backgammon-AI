import numpy as np
import gym
import time
from itertools import count
import random
from gym_backgammon.envs.backgammon import WHITE, BLACK, COLORS, TOKEN
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import models
import utils
import argparse

parser = argparse.ArgumentParser(description="Training our model")

parser.add_argument('-e', '--episodes', dest='n_episodes', type=int, default = 3, help='Number of Episodes')
parser.add_argument('-m', '--model', dest='model_checkpoint', type=str, default = 'old', help='Load in trained model.')


args = parser.parse_args()


env = gym.make('gym_backgammon:backgammon-v0')

print(args.model_checkpoint)
def main():
    
    if args.model_checkpoint != 'new':
        print('loading model...')
        network = torch.load('./{}'.format(args.model_checkpoint) + '.pth')
        print('loaded model :', args.model_checkpoint)
        
    elif args.model_checkpoint == 'new':
        #start again from scratch.
        print('new model...')
        network = utils.Network() #just incase
        for p in network.parameters():
            nn.init.zeros_(p)
        
        
    
    agent_white = utils.TD_BG_Agent(env, WHITE, network)
    agent_black = utils.TD_BG_Agent(env, BLACK, network)
    
    losses = utils.train(agent_white, agent_black, env, n_episodes=args.n_episodes)
    
    #Save the model.
    torch.save(agent_white.network, './{}'.format(args.model_checkpoint) + str(time.time()) + '.pth')
    
    fig, ax = plt.subplots()
    
    ax.plot(losses)

    plt.savefig('losses.png')
    plt.show()


if __name__ == '__main__':
    
    main()
