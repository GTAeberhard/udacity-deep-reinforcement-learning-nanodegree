# Project 1: Navigation

## Introduction

For this project, an agent is trained to naviate a large, square world with the goal of collecting bananas.

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.
Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around
agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete
actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100
consecutive episodes.

## Installation

After checking out the repository and navigating to this project's root folder, a simple setup script is provided that
will install of the required Python dependencies and download the required Unity Banana environments
from Udacity. As a pre-requisite, Python 3 must be installed on your system.

```
$ pip install .
$ python ./environment/download_environment.py
```

Once this command executes, it should be possible to immediately run the application as described below.
Note that it is recommended to setup a [`virtualenv`](https://docs.python-guide.org/dev/virtualenvs/) environment for
this project so that only the required dependencies are installed and your system's Python setup remains untouched.

## User Instructions

The main application for running the agent in the environment is executed using the Python script
`banana_navigation.py`.

```
$ python banana_navigation.py
```

By default, running the application without any parameters will run the application in inference mode with the DQN
weights loaded from `weights.pth`. A single episode of the environment is run with the final score being output on the
console.

To run the application in headless mode, i.e. without the visualization, use the `--headless` argument.

```
$ python banana_navigation.py --headless
```

To see all possible options for the application, run the help command:

```
$ python banana_navigation.py --help
```

### Inference Mode

The default mode of the application is to do a single episode of the environment in inference mode, which simply runs
the environment with the currently trained agent. By default, the neural network weights from the file `weights.pth`
are loaded. If this file does not exist, the neural network is simply initialized to random values.

The following command can be used to specify your own neural network parameters, which is useful for seeing the
inference performance of differently trained networks.

```
$ python banana_navigation.py --load_parameters /path/to/parameters/weights.ptn
```

### Training Mode

In order to train a new agent, the training mode of the application must be activated by running with the following
command:

```
$ python banana_navigation.py --mode train
```

This will train a basic Deep Q-Network to solve the banana environment and save the weights to the file `weights.pth`
once an average score of 13 over 100 episode is reached.  The `--save_parameters` command can be used to specify your
own output file name for the trained weights.

The `-o` option can be used to activate different extensions to the basic DQN algorithm. For example, in order to train
a model using Double Q-Learning and prioritized experience replay, run the following:

```
$ python banana_navigation.py --mode train -o double prioritzed_replay dueling
```

Currently supported extensions:
* [Double Q-Learning](https://arxiv.org/abs/1509.06461)
* [Dueling Deep Q-Network](https://arxiv.org/abs/1511.06581)
* [Prioritzed Experience Replay](https://arxiv.org/abs/1511.05952)

In order to see the training results (during and after training), the episode scores as well as the mean score over
the last 100 episodes is published to a TensorBoard compatible log directory. In order to view the results on
TensorBoard, run

```
$ tensorboard --logdir runs
```

In order to specficy a custom name for the TensorBoard run, use the `--name` option.

### Manual Mode

Want to know how well you match up with the artificial intelligence agent? Then try out manual mode, where you can have
a go at collecting some bananas in the environment yourself. In order to launch manual mode, run the application with
the following command:

```
$ python banana_navigation.py --mode manual
```

The following commands can be used to control the agent:
- **`W`** - move forward. (Note: when playing the game, the agent will move forward, if you don't select a different
            action in time, so you can also think of this action as the "do nothing" action.)
- **`S`** - move backward.
- **`A`** - turn left.
- **`D`** - turn right.

Note that the manual mode will not run when running the application is in headless mode.

Good Luck!

## Results

For detailed results on how this environment was solved using various Deep Q-Learning methods, please refer to the
[Project Report](Report.md).

Note that the starting point for the implementation in this repository, the base DQN implementation from the
Udacity Deep Reinforcement Learning Nanodegree exercises were used, in particular the exercise for solving the Lunar
Lander environment, which can be found [here](https://github.com/udacity/deep-reinforcement-learning/tree/master/dqn).