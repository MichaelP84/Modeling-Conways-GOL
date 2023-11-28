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

The limitations of predicting complex systems makes Conway's Game of Life a captivating study. My objective was to abstract the rules of this simulation into a different computationally irreducible system. Or, in other words, get a nueral network to learn the update rules of the simulation. During this project, the model's were trained on game data of a 10x10 grid as it played out, where after every game, the model was trained on a random sample of 1000 previous states.

![](https://github.com/MichaelP84/Modeling-Conways-GOL/blob/main/git_resources/GAME.gif)

The model predictions on this initial game state will be used for visually comparing results.

### Baseline

What I will be comparing my results to is a baseline model, where the prediction for the next state is the last state of the simulation.

Over 100 trials, the accuracy of the baseline was 87%

### Convolutional Nueral Network

Over 100 trials, the accuracy of the CNN was 99.88%

![](https://github.com/MichaelP84/Modeling-Conways-GOL/blob/main/git_resources/CNN.gif)

The raw output values of the models are seen in the GIF via the numbers, and these numbers are reflected as right and wrong as green and red, where the magnitude of correctness is proportional to the virbance of said color.

### Linear Network

Over 100 trials, the accuracy of the Linear Network was 99.81%

![](https://github.com/MichaelP84/Modeling-Conways-GOL/blob/main/git_resources/FF.gif)


### Conclusion

Both models did suprisingly well, and outperformed the baseline by about 12% to 13%. It is interesting to observe how while the CNN and Linear networks learned the underlying rules of Conway's game of life, the CNN, utilizes some compression of information/ dimension reduction during the initial convolutional layer. As a result, in the GIF for the CNN, a lot of residual values close to 0 are seen around the correct values. This contrasts the Linear model where values tend to stay at 0 when not a strong guess. This makes me hypothesize that on a larger grid size (20x20 or more), a CNN's filters would be able to better learn the rules of Conway's Game of Life. Meaning, a CNN would outperform a purely Linear model of the same size at this level. However, that will have to be the focus of a different experiment.




