import socket
import time
import random
import numpy as np
import json
from StateHandler import  StateHandler
from logger import Logger
from FA_config import *
from function_approximator import FA
from IPython import embed
import matplotlib
# matplotlib.use('QT4Agg') # Use for OSX
import matplotlib.pyplot as plt


HOST = '127.0.0.1'
PORT =  9000

ACTION_SET = ["left", "right", "down", "up", "grab", "release"]

def sendMessage(sock, msg):
    sock.send(str.encode(msg))
    BUFF_SIZE = 1440000  # 4 KiB
    data = b''
    time.sleep(0.04)
    while True:
        sock.settimeout(5.0)
        part = sock.recv(BUFF_SIZE)
        sock.settimeout(None)
        data += part
        if len(part) < BUFF_SIZE:
            #either 0 or end of data
            break
        if not part:
            break
    response = data.decode()
    return response

def getState(sock, stateFormatter, debug=False):
    response = sendMessage(sock, "observe")

    if (debug):
        print("RawState ", response)
    state = stateFormatter.formatCupsworldState(json.loads(response))
    return state

if __name__ == "__main__":
    manual = FA_MANUAL

    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "-m":
            manual = True

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((HOST, PORT))
    fa  = FA(sock, MANUAL=manual, DEBUG=FA_DEBUG)

    if (FA_READ_WEIGHTS):
        with open(FA_WEIGHTS_PATH) as f:
            data = json.load(f)

        print("Reading in weights ", data)

        for key in data:
            fa.weightsForActions[key] = np.array(data[key])

    cumulativeRewards = []
    episodeLengths = []
    startTime = time.time()
    for j in range(NUM_EPISODES):
        print("Running episode: ", j)
        sendMessage(sock, "reset")

        time.sleep(0.1)

        fa.reset()
        epLength = fa.n_step_TD()

        episodeLengths.append(epLength)
        cumulativeRewards.append(fa.cumulativeReward)

        print("terminals achieved: ", fa.numTerminalStates)


    runTime = (time.time() - startTime) / 60.
    print("Ran for {:2.2f} minutes.".format(runTime))

    # note that output.json must already exist at this point
    with open(FA_WEIGHTS_PATH, 'w') as f:
        reformatted = dict()
        for key in fa.weightsForActions:
            reformatted[key] = fa.weightsForActions[key].tolist()

        # this would place the entire output on one line
        # use json.dump(lista_items, f, indent=4) to "pretty-print" with four spaces per indent
        json.dump(reformatted, f)



    plt.scatter([i for i in range(len(cumulativeRewards))], cumulativeRewards)
    plt.title("Award Per Episode")
    plt.show()

    plt.scatter([i for i in range(len(episodeLengths))], episodeLengths)
    plt.title("Episode Length")
    plt.show()

    plt.bar(fa.actionCount.keys(), fa.actionCount.values(), 0.5, color='g')
    plt.show()
    