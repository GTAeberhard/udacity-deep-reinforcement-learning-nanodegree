# Project 3: Collaboration and Competition

## Introduction

For this project, two agents play a game of table tennis in the given environment. A reward of +0.1 is given if an agent
successfully hits the ball over the net. A reward of -0.01 is given if an agent lets the ball hit the ground/table or
if the ball goes out-of-bounds. With this reward structure, it is in both agent's interest to keep the ball in play.
The input observation state space consists of 8 variables, which represent the position and velocity of the agent's
racket and the ball. Note that each agent receives its own, local observation. The output actions are are two variables,
one representing the movement of the racket to the left and right, and the other representing the racket's upward
movement.

At the end of each episode, two scores are given, one for each agent. The episode's score is the maximum of the two
scores. In order to solve the environment, the an average episode score of +0.5 must be achieved over 100 consecutive
episodes.

## Installation

After checking out the repository, navigate to the Collaboration and Competition project's folder.

```
$ cd Project_CollaborationCompetition
```

It is recommended to setup a [`virtualenv`](https://docs.python-guide.org/dev/virtualenvs/) environment for this project
so that only the required dependencies are installed and your system's Python setup remains untouched. The following
steps will setup such an environment and activate it.

```
$ virtualenv -p /usr/bin/python3.6 venv
$ source ./venv/bin/activate
```

To deactivate the virtual environment later, simply execute the command `deactivate`.

In order to install all of the required dependencies to run the project, install the project using pip, and then run an additional script which downloads the required Unity environments.

```
$ pip install .
$ python ./environment/download_environment.py
```

It should now be possible to immediately run the application as described in the User Instructions below. To verify that
everything works, you can run the unit tests for the tennis environment, which will run the trained agents on the
tennis environment and verify that a score larger than 0.5 was achieved.

```
$ python -m pytest
```

## User Instructions

The main application for running the agent in the environment is executed using the Python script `tennis.py`.

```
$ python tennis.py
```

By default, this will run two agents that were trained with a multi-agent version of the Deterministic Deep Policy
Gradient (DDPG) algorithm for a single episode with actor and critic network weights being loaded from
`weights_actor.pth` and `weights_critic.pth`, respectively. The final score of the episode is output to the console.

To run the application in headless mode, i.e. without the visualization, use the `--headless` argument.

```
$ python tennis.py --headless
```

To see all possible options for the application, run the help command:

```
$ python tennis.py --help
```

### Inference Mode

By default, the application will run in inference mode with a network trained by the Multi-Agent Deep Deterministic
Policy Gradient algorithm and try to load the network weights from the files `weights_actor.pth` and
`weights_critic.pth`. If these files do not exist, the neural network is simply initialized to random values.

The following command can be used to specify your own neural network parameters, which is useful for seeing the
inference performance of differently trained networks.

```
$ python tennis.py --load_parameters /path/to/parameters/weights_actor.pth /path/to/parameters/weights_critic.pth
```

### Training Mode

In order to train new agents, the training mode of the application must be activated by running with the following
command:

```
$ python tennis.py --mode train
```

This will train multi-agent actor-critic agents to solve the tennis environment and save the weights of the actor and
critic networks at the end of training to the files `weights_actor.pth` and `weights_critic.pth`. Once an average score
of +0.5 over 100 consecutive episodes is reached, the environment is considered solved and additional network weights
files are saved with the suffix `_e{}` where `{}` is the episode number that solved the environment. Additionally, the
episode which had the highest score is also saved to the files `weights_max_actor.pth` and `weights_max_critic.pth`. The
`--save_parameters` command can be used to specify your own output file name for the trained actor and critic network
weights.

In order to see the training results (during and after training), the episode scores, the mean score over
the last 100 episodes, and the loss for the actor and critic networks is published to a TensorBoard compatible log
directory. In order to view the results on TensorBoard, run

```
$ tensorboard --logdir runs
```

In order to specify a custom name for the TensorBoard run, use the `--name` option.

It is possible to specify your own hyperparameters for training, including the architecture of the neural networks,
using a JSON file. Simply create a JSON file with all of the necessary hyperparameters (see
[agent/hyperparameters.py](agent/hyperparameters.py) for the complete list) and load it using

```
$ python tennis.py --hyperparameters_json /path/to/your/hyperparameters.json
```

### Manual Mode

Test your own tennis skills using the manual mode, where you can use the keyboard to control both agents during the
tennis game. Run the application in manual mode using:

```
$ python tennis.py --mode manual
```

The keyboard controls for both agents are summarized in the table below.

| Action            | Left Agent | Right Agent |
| ----------------- | ---------- | ----------- |
| Move racket left  | A          | Left Arrow  |
| Move racket right | D          | Right Arrow |
| Move racket up    | W          | Up Arrow    |
| Move racket down  | S          | Down Arrow  |

## Results

For detailed results on how this environment was solved using multi-agent Actor-Critic methods, please refer to the
[Project Report](Report.md).
