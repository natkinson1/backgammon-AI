import numpy as np
import gym
import time
from itertools import count
import random
from gym_backgammon.envs.backgammon import WHITE, BLACK, COLORS, TOKEN
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class Network(nn.Module):
    
    '''Predicts the probabilty that the player playing white is going to win.'''
    
    def __init__(self):
        super(Network, self).__init__()
        self.x_layer = nn.Sequential(nn.Linear(198, 50), nn.Sigmoid())
        self.y_layer = nn.Sequential(nn.Linear(50, 1), nn.Sigmoid())

    def forward(self, state):
        
        xh = self.x_layer(state)
        win_probability = self.y_layer(xh)
        
        return win_probability
    

class TD_BG_Agent:
    
    def __init__(self, env, player, network):
        
        self.env = env
        self.player = player
        self.network = network
        self.elig_traces = None
        self.lamda = 0.7
        self.learning_rate = 0.1
        self.gamma = 0.99
        
    def roll_dice(self):
        
        #This may need to be swapped around.
        return (-random.randint(1, 6), -random.randint(1, 6)) if self.player == WHITE\
               else (random.randint(1, 6), random.randint(1, 6))
        
    
    def best_action(self, roll):
        
        '''Choose best action'''
        
        alpha = 0.01
        actions = list(self.env.get_valid_actions(roll))
        
        values = []
        
        current_state = self.env.game.save_state()
        
        for a in actions:
            
            #Take action a
            next_state, reward, terminal, winner = self.env.step(a)
            
            #figure this out with
            val = self.network(torch.Tensor(next_state))
            
            values.append(val.detach().numpy())
            
            #Go back to where you currently are.
            self.env.game.restore_state(current_state)
        
        idx = int(np.argmax(values)) if self.player == 0 else int(np.argmin(values))
        
        return actions[idx]
    
    
    def update(self, current_prob, future_prob, reward, learning_rate):
        
        self.network.zero_grad()
        current_prob.backward()
        
        delta_t = reward + self.gamma * future_prob - current_prob
        
        with torch.no_grad():
        
            for i, weights in enumerate(self.network.parameters()):
                
                self.elig_traces[i] = self.gamma * self.lamda * self.elig_traces[i] + weights.grad
                new_weights = weights + learning_rate * delta_t * self.elig_traces[i]

                weights.copy_(new_weights)
        
        return delta_t
    
    
def train(agent_white, agent_black, env, n_episodes=1, max_time_steps=3000):
    
    '''Trains two agents which compete against one another to make a
    model really good at backgammon.'''
    
    agents = {WHITE : agent_white, BLACK : agent_black}
    agent_wins = {WHITE : 0, BLACK : 0}
    other_agent ={WHITE: BLACK, BLACK : WHITE}
    
    for episode in range(n_episodes):
        
        #Reset the environment, choose randomly who goes first
        losses = []
        agent_colour, roll, state = env.reset()
        agent = agents[agent_colour]

        #reset the eligibility traces...
        agent_white.elig_traces = [torch.zeros(weights.shape, requires_grad=False)\
                        for weights in list(agent_white.network.parameters())]
        agent_black.elig_traces = [torch.zeros(weights.shape, requires_grad=False)\
                        for weights in list(agent_black.network.parameters())]
        
#         if episode % 10000 == 0:
#             agent_white.learning_rate = agent_white.learning_rate * 0.7
#             agent_black.learning_rate = agent_black.learning_rate * 0.7
        learning_rate = 0.1 * np.exp(-1 * episode * 0.00015)
        
        for i in range(max_time_steps):
            
            if i == 0:
                pass
            else: 
                roll = agent.roll_dice()
                
            
            current_prob = agent.network(torch.Tensor(state))
            
            actions = env.get_valid_actions(roll)
            
            if len(actions) == 0:
                
                agent_colour = env.get_opponent_agent()
                agent = agents[agent_colour]
                
                continue
            
            best_action = agent.best_action(roll)
            next_state, reward, terminal, winner = env.step(best_action)
            
            future_prob = agent.network(torch.Tensor(next_state))
            
            if terminal and winner is not None:
                reward_multiplier = n_pieces_outside_home(env, other_agent[winner])
                if winner == BLACK:
                    reward = -1
                    reward_multiplier = reward_multiplier * -1
                loss = agent.update(current_prob, future_prob, reward + (reward_multiplier * 0.1), learning_rate)
                loss_step = loss / i
                losses.append(loss_step.detach().numpy())

                agent_wins[winner] += 1

                wwp = 100 * (agent_wins[WHITE] / (agent_wins[WHITE] + agent_wins[BLACK]))
                bwp = 100 - wwp


                print('Episode: {} | Winner: {} | Num White Wins: {} | Num of Black Wins: {} | White Win Percentage: {:.2f} | Black Win Percentage: {:.2f} | Total loss: {:.2f}'\
                 .format(episode, agent.player, agent_wins[WHITE], agent_wins[BLACK],\
                        wwp, bwp, np.sum(losses)))
                break
                
            else:
                loss = agent.update(current_prob, future_prob, reward, learning_rate)
                losses.append(loss.detach().numpy())
            
            agent_colour = env.get_opponent_agent()
            agent = agents[agent_colour]
            
            state = next_state
            
    return losses

def n_pieces_outside_home(env, agent_colour):

    board = env.game.board

    if agent_colour == BLACK:
        outside_pieces = board[6:]
        assert len(outside_pieces) == 18
        total = 0
        for n_pieces, colour in outside_pieces:
            if colour == BLACK:
                total += n_pieces
    
    if agent_colour == WHITE:
        outside_pieces = board[:-6]
        assert len(outside_pieces) == 18
        total = 0
        for n_pieces, colour in outside_pieces:
            if colour == WHITE:
                total += n_pieces
    
    return total
