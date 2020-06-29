# backgammon-AI

In this repository, an application has been made so you can play against yourself or a friend or an AI on the command line.

# Prerequisites

This implementation assumes that you already have anaconda installed on your system. 

## gym-backgammon

The [gym-backgammon](https://github.com/dellalibera/gym-backgammon) environment is used to train the reinforcement learning agent. I also use it to render the game.

To install the environment run:

```
git clone https://github.com/dellalibera/gym-backgammon.git
cd gym-backgammon/
pip3 install -e .
```
You will also need [Pytorch](https://github.com/pytorch/pytorch):

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

You can choose to play against another person using only human moves. Alternatively you can play against an AI.

## Human Move

If you dont have any dice to hand you can roll some virtual dice:

```
python roll_dice.py
```

To play a move of your choosing run:

```
python play -human (source) (destination)
```

*source* : The number representing where the piece is on the board.

*destination* : The number representing where you want that piece to move to.

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

## Moving a piece which has been taken.

To move a piece which is on the bar after being taken, we denote the source to be *bar*.

### Example

Say if you are player X and had 2 pieces on the bar and had rolled a 1 4:

```
python play.py -human bar 23 bar 20
```

## Moving a piece off the board

At the ending stage of a game when all your pieces are home, to take the pieces off the board the destination is:

X home board : -1 

O home board : 24

### Example

If you are player X and rolled a 6 and 4:

```
python play.py -human 5 -1 3 -1
```

If you are player O and rolled a 6 and 4:

```
python play.py -human 18 24 20 24
```

## When you can't make a move

If you roll a dice which results in you being unable to make a move run (not needed if it is the AI's turn):

```
python play.py -skip
```

## AI move

If you would like to let the AI make what it thinks the best move is given a roll:

```
python play.py -ai (roll)
```

*roll* : The roll you want the AI to make a play with.

### Example

If you rolled a 3 4 you would run:

```
python play.py -ai 3 4
```

Running this automatically updates the board with the move the AI chose. 

**Note**
 - If you roll a double you **do not** need to input 4 dice numbers. The AI will automatically make 4 moves if a double is rolled.
 - If you roll a dice that you know the AI cant make a move on. Still input this roll and the AI will recognise it cannot make a move.

# AI

The AI played 200,000 games against itself to learn a model capable of playing a good game of backgammon. The AI used a reinforcement learning algorithm to train the network.
