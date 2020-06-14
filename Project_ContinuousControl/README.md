# Project 2: Continuous Control

## Introduction

For this project, an agent is trained to move a double-jointed robotic arm to a goal position, represented by a moving sphere in space. A reward of +0.1 is is provided for every time step that the end of the arm is within the target goal position sphere. The input observation for the agent consists of 33 variables which correspond to position, rotation, velocity, and angular velocity of the robot's arm. The four action outputs corresond to the torque which can be applied to the two joints of the arm. The output actions are represented as a number between -1 and 1, making this a continuous action-space reinforcement learning problem.

Two versions of the environment are given: one with a single agent, another with multiple agents (20 to be exact).  In order to solve the environment, a score of +30 must be achieved for a single episode (where for the multiagent environment, the average over all 20 agents per episode is used).

## Installation

Check out the repository and navigate into this project's root folder (`cd Project_ContinuousControl`). A simple install script is provided which installs all necessary Python dependencies and downloads the required Unity environments.  As a pre-requisite, Python 3 must be installed on your system.

```
$ pip install -e .
```

Once this command executes, it should be possible to immediately run the application as described below. Note that it is recommended to setup a [`virtualenv`](https://docs.python-guide.org/dev/virtualenvs/) environment for this project so that only the required dependencies are installed and your system's Python setup remains untouched.

## User Instructions

The main application for running the agent in the environment is executed using the Python script
`reacher.py`.

```
$ python reacher.py
```

By default, this will run an agent that was trained with the Deterministic Deep Policy Gradient (DDPG) algorithm for a single episode with actor and critic network weights being loaded from `actor_weights.pth' and 'critic_weights.pth', respectively. The final score of the episode is output to the console.

To run the application in headless mode, i.e. without the visualization, use the `--headless` argument.

```
$ python reacher.py --headless
```

To see all possible options for the application, run the help command:

```
$ python reacher.py --help
```

### Inference Mode

### Training Mode

## Results

For detailed results on how this environment was solved using Actor-Critic methods, please refer to the [Project Report](Report.md).