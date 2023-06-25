# Modeling-Conways-Game-of-Life
Modeling cellular automata/re-representing a computationally irreducible system.

![](https://github.com/MichaelP84/Modeling-Conways-GOL/blob/main/git_resources/ConwayGif.gif)


## The Game

Conway's Game of Life is a cellular automaton devised by mathematician John Conway. It consists of a grid of cells, each of which can be in one of two states: alive or dead. The game evolves in discrete steps, following a set of simple rules. At each step, the state of each cell is determined by the states of its eight neighboring cells. The rules are as follows:

1. Any live cell with fewer than two live neighbors dies, as if by underpopulation.
2. Any live cell with two or three live neighbors lives on to the next generation.
3. Any live cell with more than three live neighbors dies, as if by overpopulation.
4. Any dead cell with exactly three live neighbors becomes a live cell, as if by reproduction.

These rules create fascinating patterns and behaviors in the grid, from stable configurations to oscillators and even self-replicating structures. The Game of Life is not a traditional game with players, but rather a simulation that explores the emergence of complex patterns from simple rules.

Conway's Game of Life is a computationally irreducible system, meaning there is no known shortcut to predict its future states without simulating each step. The game's complexity arises from simple rules and the exponential growth of configurations. Interactions between patterns further amplify unpredictability. In this core way, Conway's Game of Life is much like the life we experience. Although we occasinally find computational shortcuts as we build upon our model of the world, the particles and forces of nature can be seen as the primordial computationally irreducible system. 



## The Model

The limitations of predicting complex systems makes Conway's Game of Life a captivating study. My objective was to abstract the rules of this simulation into a different computationally irreducible system. Or, in other words, get a nueral network to learn the update rules of the simulation.

What I will be comparing my results to is a baseline model, where the prediction for the next state is the last state of the simulation.

### Baseline



### CNN



### Fully Connected




