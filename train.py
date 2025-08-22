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
import os

parser = argparse.ArgumentParser(description="Training our model")

parser.add_argument('-e', '--episodes', dest='n_episodes', type=int, default = 3, help='Number of Episodes')
parser.add_argument('-m', '--model', dest='model_checkpoint', type=str, default = 'old', help='Load in trained model.')
parser.add_argument('-o', '--output_file', dest='dest_file', type=str, default = 'new_model.pth', help='Save new file location')


args = parser.parse_args()


env = gym.make('gym_backgammon:backgammon-v0')

print(args.model_checkpoint)
def main():
    
    if args.model_checkpoint in [f for f in os.listdir("./") if f.endswith(".pth")]:
        print('loading model...')
        network = torch.load('./{}'.format(args.model_checkpoint), weights_only=False)
        print('loaded model :', args.model_checkpoint)
        
    elif args.model_checkpoint != 'old':
        #start again from scratch.
        print('new model...')
        network = utils.Network() #just incase
        for p in network.parameters():
            nn.init.zeros_(p)
        
        
    
    agent_white = utils.TD_BG_Agent(env, WHITE, network)
    agent_black = utils.TD_BG_Agent(env, BLACK, network)
    
    losses = utils.train(agent_white, agent_black, env, n_episodes=args.n_episodes)
    
    #Save the model.
    torch.save(agent_white.network, './{}'.format(args.dest_file))
    
    fig, ax = plt.subplots()
    
    ax.plot(losses)

    plt.savefig('losses.png')


if __name__ == '__main__':
    
    main()
