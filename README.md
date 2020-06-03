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

To start a game run the script play.py with tag -begin:

```
python play.py -begin
```
This should display an ascii-art representation of a backgammon board.

```
| 12 | 13 | 14 | 15 | 16 | 17 | BAR | 18 | 19 | 20 | 21 | 22 | 23 | OFF |
|--------Outer Board----------|     |-------P=O Home Board--------|     |
|  X |    |    |    |  O |    |     |  O |    |    |    |    |  X |     |
|  X |    |    |    |  O |    |     |  O |    |    |    |    |  X |     |
|  X |    |    |    |  O |    |     |  O |    |    |    |    |    |     |
|  X |    |    |    |    |    |     |  O |    |    |    |    |    |     |
|  X |    |    |    |    |    |     |  O |    |    |    |    |    |     |
|-----------------------------|     |-----------------------------|     |
|  O |    |    |    |    |    |     |  X |    |    |    |    |    |     |
|  O |    |    |    |    |    |     |  X |    |    |    |    |    |     |
|  O |    |    |    |  X |    |     |  X |    |    |    |    |    |     |
|  O |    |    |    |  X |    |     |  X |    |    |    |    |  O |     |
|  O |    |    |    |  X |    |     |  X |    |    |    |    |  O |     |
|--------Outer Board----------|     |-------P=X Home Board--------|     |
| 11 | 10 |  9 |  8 |  7 |  6 | BAR |  5 |  4 |  3 |  2 |  1 |  0 | OFF |
```

## Human Move

If you dont have any dice to hand you can run roll_dice.py to make a randomly rolled dice.

```
python roll_dice.py
```

To play a move of your choosing run:

```
python play -human (source) (destination)
```

source : The number representing where the piece is on the board.
destination : The number representing where you want that piece to move too.

### Example

Let's say I roll a 6 1 and I am moving the X pieces.

The desired board position I want is:

```
| 12 | 13 | 14 | 15 | 16 | 17 | BAR | 18 | 19 | 20 | 21 | 22 | 23 | OFF |
|--------Outer Board----------|     |-------P=O Home Board--------|     |
|  X |    |    |    |  O |    |     |  O |    |    |    |    |  X |     |
|  X |    |    |    |  O |    |     |  O |    |    |    |    |  X |     |
|  X |    |    |    |  O |    |     |  O |    |    |    |    |    |     |
|  X |    |    |    |    |    |     |  O |    |    |    |    |    |     |
|    |    |    |    |    |    |     |  O |    |    |    |    |    |     |
|-----------------------------|     |-----------------------------|     |
|  O |    |    |    |    |    |     |  X |    |    |    |    |    |     |
|  O |    |    |    |    |    |     |  X |    |    |    |    |    |     |
|  O |    |    |    |    |    |     |  X |    |    |    |    |    |     |
|  O |    |    |    |  X |  X |     |  X |    |    |    |    |  O |     |
|  O |    |    |    |  X |  X |     |  X |    |    |    |    |  O |     |
|--------Outer Board----------|     |-------P=X Home Board--------|     |
| 11 | 10 |  9 |  8 |  7 |  6 | BAR |  5 |  4 |  3 |  2 |  1 |  0 | OFF |
```

I would run:

```
python play.py -human 12 6 7 6
```

If I rolled 6 6, I have 4 moves so I would run:

```
python play.py -human 12 6 12 6 23 17 23 17
```

This would give me the board:

```
| 12 | 13 | 14 | 15 | 16 | 17 | BAR | 18 | 19 | 20 | 21 | 22 | 23 | OFF |
|--------Outer Board----------|     |-------P=O Home Board--------|     |
|  X |    |    |    |  O |  X |     |  O |    |    |    |    |    |     |
|  X |    |    |    |  O |  X |     |  O |    |    |    |    |    |     |
|  X |    |    |    |  O |    |     |  O |    |    |    |    |    |     |
|    |    |    |    |    |    |     |  O |    |    |    |    |    |     |
|    |    |    |    |    |    |     |  O |    |    |    |    |    |     |
|-----------------------------|     |-----------------------------|     |
|  O |    |    |    |    |    |     |  X |    |    |    |    |    |     |
|  O |    |    |    |    |    |     |  X |    |    |    |    |    |     |
|  O |    |    |    |  X |    |     |  X |    |    |    |    |    |     |
|  O |    |    |    |  X |  X |     |  X |    |    |    |    |  O |     |
|  O |    |    |    |  X |  X |     |  X |    |    |    |    |  O |     |
|--------Outer Board----------|     |-------P=X Home Board--------|     |
| 11 | 10 |  9 |  8 |  7 |  6 | BAR |  5 |  4 |  3 |  2 |  1 |  0 | OFF |
```

There is no contraint on the number of pieces I can move in 1 turn.

## AI move

If you would like to let the AI make a move



