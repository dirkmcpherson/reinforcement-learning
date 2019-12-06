import numpy as np
import learner
import subjective
import env
import action
from IPython import embed
import matplotlib
matplotlib.use("agg")

import matplotlib.pyplot as plt
import time
import sys


def main():
    pass

def reset(arm):
    arm.reset()


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    save = False
    load = False
    if (len(sys.argv) > 2):
        val = sys.argv[2]
        if (val == 0):
            pass
        elif (val == 1):
            save = True
        elif (val == 2):
            load = True
        elif (val == 3):
            save = True
            load = True

    dynaQ = True
    descriptionString = "dynaQ" if dynaQ else "basic"
    numExperiments = 5
    allEpisodes = []
    for i in range(numExperiments):
        plt.figure(i)
        numEpisodes = 100
        action_resolution = 1 # how many timesteps between updates?
        step_count = 0

        arm = env.ArmEnv()
        o = subjective.Subjective(arm)
        l = learner.DynaQLearner()
        if (load):
            print("Loading previously learned values and model")
            l.load()

        actionIdx = action.getRandomActionIdx()

        indexCount = [0 for entry in action.ActionSet]
        AchievedGoal = False # have we ever achieved the goal

        stepstogoal = 0
        allSteps = []
        while(True):
            startTime = time.time()
            arm.render() # render the arm visually
            # print("time0 ", time.time() - startTime)

            TakeStep = (step_count % action_resolution) == 0
            if TakeStep:
                state = arm.get_state()
                actionIdx = l.sampleAction(state)

                # actionIdx = max(0, min(2, int(input("enter action idx"))))

                indexCount[actionIdx] += 1
            # print("time1 ", time.time() - startTime)

            newState, reward, goalAchieved = arm.step(action.ActionSet[actionIdx]) # select an action based on the current policy
            # print("newState ", newState)
            # print("reward ", reward)
            # allRewards.append(reward)
            # print("time2 ", time.time() - startTime)

            if (goalAchieved or TakeStep):
                l.update(state, actionIdx, newState, reward, updateModel=True)
                # print("time2 ", time.time())

            # print("time3 ", time.time() - startTime)
            # if (step_count > 1 and step_count % 1000 == 0):
            #     embed()

            # feedback = o.evaluateEnvironment(arm) # Oracle observes the environment so check if it has feedback
            # if feedback:
            #     print("Gave feedback ", feedback.valence)
            #     l.acceptFeedback(state)

            if (goalAchieved):
                if not AchievedGoal:
                    print("First goal achieved.")
                    AchievedGoal = True
                reset(arm)
                allSteps.append(stepstogoal)
                # print("Episode complete. ", stepstogoal)
                # print("    mean: ", np.mean(allSteps))
                # print("    stedev: ", np.std(allSteps))
                stepstogoal = 0
                numEpisodes -= 1
                if (numEpisodes <= 0):
                    break
            elif (AchievedGoal): # dont care about updates until we have some reward to propagate
                if (dynaQ):
                    l.modelUpdate(arm)
                else:
                    pass

            # if (step_count % 100000 == 0):
            #     # print("All rewards: ", np.sum(allRewards))
            #     for entry in l.Q:
            #         print("{} - {}".format(entry, l.Q[entry]))
            #     print("-------------------------")
            #     allRewards.clear()
            
            step_count += 1
            stepstogoal += 1
            # print("duration: ", time.time() - startTime)

        allEpisodes.append(allSteps)
        print("All data:")
        for entry in allSteps:
            print("      ", entry)
        print("Final Mean: ", np.mean(allSteps))
        print("Final stdev: ", np.std(allSteps))

        plt.plot(allSteps)
        plt.savefig("experiment-{}-{}".format(descriptionString,i))

    allMeans = np.mean([np.mean(entry) for entry in allEpisodes])
    print("allMeans: ", allMeans)

    episodicMean = []
    for i in range(len(allEpisodes[0])):
        total = [entry[i] for entry in allEpisodes]
        episodicMean.append(np.mean(total))

    plt.figure(9)
    plt.plot(episodicMean)
    plt.savefig("episodicMean-{}-{}".format(descriptionString, i))

    np.save("analytics", np.array(allEpisodes))
    if (save):
        print("Saving learned values and model")
        l.save()
    # print("what")
    # # plt.figure(0)
    # print("what1")
    # plt.subplot(111)
    # print("what2")
    # plt.plot([i for i in range(len(allRewards))], allRewards)
    # print("what3")
    # plt.show()
    # print("Training complete.")