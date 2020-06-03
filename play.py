#to be imported into a python kernel.
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import utils
import gym
import warnings
warnings.filterwarnings('ignore')
import argparse
import pickle

parser = argparse.ArgumentParser(description="Play Backgammon")

parser.add_argument('-begin',
                    dest='begin',
                    action='store_true',
                    help='Start the game')
parser.add_argument('-starting_player',
                    dest='colour',
                    type=str,
                    default='X', 
                    help='Define which player starts first (Either X or O)')
parser.add_argument('-ai',
                    dest='ai',
                    type=int,
                    default=None,
                    nargs='+',
                    help='Define what you have rolled, so the agent can choose the optimal move')
parser.add_argument('-human',
                    dest='human',
                    default=None,
                    nargs='+',
                    help='Define what move the opponent has done.')
parser.add_argument('-skip',
                    dest='skip',
                    action='store_true',
                    help='Parse if a human player cannot go.')

args = parser.parse_args()

trained_model = torch.load('./model.pth')

player = {0 : 'Red', 1 : 'Black'}
pieces = {0 : 'X', 1 : 'O'}
opp_pieces = {'X' : 0, 'O' : 1}

env = gym.make('gym_backgammon:backgammon-v0')


def save(state, agent_colour, only_agent_colour=False):
    if only_agent_colour == False:
        with open("state.pickle", "wb") as f:
            pickle.dump(state, f)
        
    with open("agent_colour.pickle", "wb") as f:
        pickle.dump(agent_colour, f)
        
        
def load():
    
    state = pickle.load(open("state.pickle", "rb"))
    agent_colour = pickle.load(open("agent_colour.pickle", "rb"))
    
    return state, agent_colour

def start_game():
    
    agent_colour, roll, state = env.reset()
    
    while opp_pieces[args.colour] != agent_colour:
    
        agent_colour, roll, state = env.reset()
    
    print()
    env.render()
    print('{} player to start.'.format(pieces[agent_colour]))
    
    save_state = env.game.save_state()
    
    save(save_state, agent_colour)
    
def ai_move():
    
    #get back to the current state:
    state, agent_colour = load()
    env.game.restore_state(state)
    env.current_agent = agent_colour
    
    #load the agent in the environment
    network = torch.load('./model.pth')
    agent = utils.TD_BG_Agent(env, agent_colour, network)
    
    
    #sort the roll depending on whos turn it is.
    roll = np.array(args.ai)
    roll = -roll if agent_colour == 1 else roll
    
    action_set = env.get_valid_actions(tuple(roll))
    
    if len(action_set) == 0:
        print()
        print('You can not make a move given your roll!')
        print()
        env.render()
        
        next_agent_colour = env.get_opponent_agent()
        
        save_state = env.game.save_state()

        save(save_state, next_agent_colour)
        
    else:
        action = agent.best_action(tuple(-roll))
        next_state, _, terminal, _ = env.step(action)
        
        if terminal:
            print()
            print('AI has won!')
        else:
            print()
            print('Estimated probability AI is going to win: {}'.format(agent.network(torch.Tensor(next_state))[0]))

        #observe action
        print()
        print('AI chose action: {}'.format(action))
        print()
        env.render()

        next_agent_colour = env.get_opponent_agent()
        print('{} player to move.'.format(pieces[next_agent_colour]))
        
        save_state = env.game.save_state()

        save(save_state, next_agent_colour)

def human_move():
    
    '''Play a move'''
    
    #get back to the current state:
    state, agent_colour = load()
    env.game.restore_state(state)
    env.current_agent = agent_colour
    
    #play the move that the human chose.
    action = args.human
    
    #For the case when you want to move pieces from the bar onto the board.
    for i in range(len(action)):
        try:
            action[i] = int(action[i])
        except:
            pass
    
    action = tuple(map(tuple, np.array(action, dtype=object).reshape(-1,2)))
    
    try:
        next_state, _, terminal, _ = env.step(action)
    except AssertionError:
        raise ValueError('Action {} is invalid. (You are moving the ({}) pieces.)'.format(action, pieces[agent_colour]))
    
    print()
    env.render()

    if terminal:
        print('{} player won!'.format(pieces[agent_colour]))
    else:
    
        next_agent_colour = env.get_opponent_agent()
        
        print('{} player to move.'.format(pieces[next_agent_colour]))

        save_state = env.game.save_state()

        save(save_state, next_agent_colour)
    
    
if __name__ == '__main__':
    
    if args.begin:
        start_game()
    elif args.ai is not None:
        ai_move()
    elif args.skip:
        
        state, agent_colour = load()
        env.game.restore_state(state)
        env.current_agent = agent_colour
        
        next_agent_colour = env.get_opponent_agent()
        print()
        print('{} player has skipped their go.'.format(pieces[agent_colour]))
        print()
        env.render()
        print('{} player to move.'.format(pieces[next_agent_colour]))
        save(state, next_agent_colour, only_agent_colour=True)
    else:
        human_move()

    
