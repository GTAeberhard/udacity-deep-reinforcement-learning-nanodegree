DNN_ARCHITECTURE = [256, 128]

# General parameters
LEARNING_RATE = 1e-3
GAMMA = 0.99
GRADIENT_CLIPPING_MAX = 1.0

# Parameters specific to DDPG agent
LEARNING_RATE_CRITIC = 1e-4
BATCH_SIZE = 128
BUFFER_SIZE = int(1e6)
TAU = 1e-3
WEIGHT_DECAY = 0.0001
STEPS_BETWEEN_LEARNING = 20
LEARNING_ITERATIONS = 10
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 1e-6

# Parameters specific to A2C agent
BELLMAN_STEPS = 5
ENTROPY_COEFFICIENT = 0.01