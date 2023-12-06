# Decomposition of Finite Normal Form Games Based on Combinatorial Hodge Theory

## Notes
The implementation is based on
> **Github repository** <br>
>[candogan-games-decomposition](https://github.com/davidelegacci/candogan-games-decomposition) by [davidelegacci](https://github.com/davidelegacci).

In this version, I made the following changes:
- reduced the code to the decomposition only
- improved runtime (by allowing the game structure to be saved/imported)
- restructered the project such that it can be installed as a python package and reused in other projects

The method is based on the following paper:
> **Flows and Decompositions of Games: Harmonic and Potential Games** <br>
> Ozan Candogan, Ishai Menache, Asuman Ozdaglar and Pablo A. Parril <br>
> *Mathematics of Operations Research, 2011*, [[Link](https://www.jstor.org/stable/23012338)]
>   

## Example
Assume we have a game with 2 agents who have 3 actions each. Then we first create the game structure:
```python
game = Game(n_actions=[4, 4], load_save=True)
```
This create the response graph and linear mappings we need for the decomposition. For larger instances this might take a while. With the keyword `load_save` we allow to save these linear maps and load them, if they were saved before. You can also define a `path` that determines where the matrices are stored/loaded. <br>

After that we can compute the decomposition for a given matrix game, .e.g, the Chicken game.
If they payoff is given by payoff matrices, we use
```python
import numpy as np
payoff_matrices = [ 
    np.array([[0, -1], [1, -100]]), 
    np.array([[0, 1], [-1, -100]])
    ]
game.compute_decomposition_matrix(payoff_matrices)
```
If the payoff is instead given in form of a payoff vector, we can directly use the decomposition method:
```python
payoff_vector = [0, -1, 1, -100, 0, 1, -1, -100] 
game.compute_decomposition(payoff_vector)
```
The result is saved in form of matrices
```python
game.payoff_matrices_P # payoff matrices for potential component
game.payoff_matrices_H # payoff matrices for harmonic component
game.payoff_matrices_N # payoff matrices for nonstrategic component
```
We also compute a metric, which denotes the potentialness of a game and which is defined by 

$$ \text{Potentialness} = \dfrac{\Vert uP \Vert_2}{\Vert uP \Vert_2 + \Vert uH \Vert_2} $$

where $uP$ and $uH$ are the payoff vectors of the potential and harmonic components. In the code you can access the metric by

```python
game.potentialness     # value of metric for potentialness
```
Using the components of the potential and the harmonic components, we can construct a new matrix game with a predefined potentialness by considering a respective convex combination of both components (if they are nonzero). This can be done in the following way
```python
new_payoff_matrices = game.create_game_potentialness(potentialness=0.5)
```
You can find this and additional examples also in the [notebooks](./notebooks/).

## Setup

Note: These setup instructions assume a Linux-based OS and uses python 3.11 (or higher).

Install virtualenv (or whatever you prefer for virtual envs)
```
sudo apt-get install virtualenv
```
Create a virtual environment with virtual env (you can also choose your own name)
```
virtualenv -p python3 venv
```
You can specify the python version for the virtual environment via the -p flag. 
Note that this version already needs to be installed on the system (e.g. `virtualenv - p python3 venv` uses the 
standard python3 version from the system).

activate the environment with
```
source ./venv/bin/activate
```

Install all requirements

```
pip install -r requirements.txt
```

Install the decomposition package.

```
pip install -e .
```

You can also run "pip install ." if you don't want to edit the code. The "-e" flag ensures that pip does not copy the code but uses the editable files instead.


### Install pre-commit hooks (for development)
Install pre-commit hooks for your project

```
pre-commit install
```
Verify by running on all files:
```
pre-commit run --all-files
```

For more information see https://pre-commit.com/.
