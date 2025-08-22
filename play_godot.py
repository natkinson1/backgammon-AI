#to be imported into a python kernel.
import numpy as np
import torch
import torch.nn as nn
import utils
import gym
import warnings
warnings.filterwarnings('ignore')
import argparse
import pickle
import ast
from collections import namedtuple

parser = argparse.ArgumentParser(description="Play Backgammon")


parser.add_argument('-move',
                    dest='ai',
                    type=str,
                    default=None,
                    help='Define what you have rolled, so the agent can choose the optimal move')
parser.add_argument('-colour',
                    dest='agent_colour',
                    type=str,
                    help='The colour of the player')
parser.add_argument('-board',
                    dest='board',
                    type=str,
                    help='The state of the board')
parser.add_argument('-bar',
                    dest='bar',
                    type=str,
                    help='The colour of the player')
parser.add_argument('-off',
                    dest='off',
                    type=str,
                    help='The colour of the player')
parser.add_argument('-player_positions',
                    dest='player_positions',
                    type=str,
                    help='The colour of the player')

args = parser.parse_args()

env = gym.make('gym_backgammon:backgammon-v0')
BackgammonState = namedtuple('BackgammonState', ['board', 'bar', 'off', 'players_positions'])

def clean_value(x):
    if isinstance(x, np.generic):  # NumPy scalar (e.g., np.int64, np.float32)
        return x.item()
    return x  # Native int, float, str, etc.

def ai_move_godot():
    
    #get back to the current state:
    # state, agent_colour = load()
    
    _, _, _ = env.reset()
    agent_colour_dict = {"black": 1, "white" : 0}

    agent_colour = agent_colour_dict[args.agent_colour]
    board = ast.literal_eval(args.board)
    bar = ast.literal_eval(args.bar)
    off = ast.literal_eval(args.off)
    player_positions = ast.literal_eval(args.player_positions)

    state = BackgammonState(board=board, bar=bar, off=off, players_positions=player_positions)
    
    env.game.restore_state(state)
    # env.current_agent = agent_colour
    env.current_agent = agent_colour_dict[args.agent_colour]
    
    #load the agent in the environment
    network = torch.load('./model_300000_games_05082025.pth', weights_only=False)
    agent = utils.TD_BG_Agent(env, agent_colour, network)
    
    
    #sort the roll depending on whos turn it is.
    roll = args.ai
    roll = roll.split(" ")
    roll = [int(x) for x in roll]
    roll = np.array(roll)
    roll = -roll if agent_colour == 1 else roll
    
    action_set = env.get_valid_actions(tuple(roll))
    
    if len(action_set) == 0:
        return []
        
    else:
        action = agent.best_action(tuple(roll))
        clean_data = [(clean_value(a), clean_value(b)) for a, b in action]

        print(clean_data)
    
    
if __name__ == '__main__':
    
    ai_move_godot()
    

    
