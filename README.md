# backgammon-AI
In this repository, an application has been made so you can play against yourself or a friend or an AI. The AI played 100000 games against itself to learn a model capable of playing a good game of backgammon. The AI used the TD($\lambda$) algorithm to train the network.

# Prerequisite

This implementation assumes that you already have anaconda installed on your system. 

## gym-backgammon

This is the [gym-backgammon](https://github.com/dellalibera/gym-backgammon) environment used to train the reinforcement learning agent. I also use it to render the game.

To install the environment run:

```
git clone https://github.com/dellalibera/gym-backgammon.git
cd gym-backgammon/
pip3 install -e .
```
You will also need [Pytorch](https://github.com/pytorch/pytorch)

```
pip install torch torchvision
```
# Play the Game

To start a game run:

```
python play.py -begin
```
This should display an ascii-art representation of a backgammon board.

## Human Move





