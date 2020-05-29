DNN_ARCHITECTURE = [64, 64]

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LEARNING_RATE = 5e-4    # learning rate
UPDATE_EVERY = 4       # how often to update the network
ALPHA = 0.6
INITIAL_BETA = 0.4
BETA_STEPS = 500
