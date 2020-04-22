'''
Author: James Staley
email: james.staley625703@tufts.edu
March 2020 :(
'''

# State Formatter configuration
WORLD_RELATIVE_TO_GRIPPER = False
PICKUP_AGENT = False
GOTO_AGENT = True

ONE_HOT_GRIPPER_STATE = True
USE_POLYNOMIAL_FEATURES = True

PICKUP_AGENT_HEIGHT = 0.7
GOTO_AGENT_DISTANCE = 0.15

DEFAULT_REWARD = -1


# Training configuration
NUM_EPISODES = 2000

# FA config
FA_LOG_LEVEL = 0
FA_MANUAL = False
FA_DEBUG = False

FA_WEIGHTS_PATH = './output.json'
FA_READ_WEIGHTS = False
FA_SAVE_WEIGHTS = True

FA_N_STEP = 20
FA_MAX_ITERATIONS = 50
FA_GAMMA = 0.5
FA_ALPHA = 0.1
