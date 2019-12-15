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
import random

DEBUG = False

def main():
    pass

def reset(arm):
    arm.reset()

def inputAndEmbed():
    char = input("Enter input...")
    if (char == 'e'):
        embed()

def printNonzeroEntries(l):
    for i in range (l.Q.shape[0]):
        vals = l.Q[i]
        if (np.sum(vals) > 0):
            print(""+str(i)+" "+str(vals))


def runAll(TotalRunEpisodes, numExperiments, dynaQ):
    descriptionString = "dynaQ" if dynaQ else "basic"
    allEpisodes = []
    for i in range(numExperiments):
        plt.figure(i)

        # TotalRunEpisodes = TotalRunEpisodes * (2 ** i)
        numEpisodes = TotalRunEpisodes

        step_count = 0

        arm = env.ArmEnv()
        l = learner.DynaQLearner()
        o = subjective.Subjective(l)
        if (load):
            print("Loading previously learned values and model")
            l.load()

        actionIdx = action.getRandomActionIdx()

        indexCount = [0 for entry in action.ActionSet]
        AchievedGoal = False # have we ever achieved the goal

        stepstogoal = 0
        allSteps = []
        while(True):
            t0 = time.time()
            arm.render() # render the arm visually
            # print("render took: {}".format(time.time() - t0))
            # t0 = time.time()

            state = arm.get_state()
            actionIdx = l.sampleAction(state)
            # indexCount[actionIdx] += 1

            
            newState, reward, goalAchieved = arm.step(action.ActionSet[actionIdx]) # select an action based on the current policy

            l.update(state, actionIdx, newState, reward, updateModel=True)
            # print("q update took: {}".format(time.time() - t0))
            # t0 = time.time()


            if (DEBUG):# and numEpisodes <= np.floor(TotalRunEpisodes/2.)):
                print("--s:{}-{} a{} sp:{}-{} r:{}--".format(state, l.tileRepresentation(state), actionIdx, newState, l.tileRepresentation(newState), reward))
                printNonzeroEntries(l)
                inputAndEmbed()

            # o.evaluateEnvironment()
            # if (numEpisodes < (3*TotalRunEpisodes/4.) and random.random() > 0.95):
            if (random.random() > 0.95):
                o.evaluateEnvironment()

            if (goalAchieved):
                if not AchievedGoal:
                    print("First goal achieved.")
                    AchievedGoal = True
                print("Achieved goal at {}".format(newState))
                reset(arm)
                allSteps.append(stepstogoal)
                # print("Episode complete. ", stepstogoal)
                # print("    mean: ", np.mean(allSteps))
                # print("    stedev: ", np.std(allSteps))
                stepstogoal = 0
                numEpisodes -= 1
                if (numEpisodes <= 0):
                    break
                # elif (numEpisodes == np.floor(TotalRunEpisodes/2.)):
                #     embed()
            elif (AchievedGoal): # dont care about updates until we have some reward to propagate
                if (dynaQ):
                    t0 = time.time()
                    l.modelUpdate(arm)
                    # print("modelUpdate took: {}".format(time.time() - t0))

                    if (DEBUG and numEpisodes <= np.floor(TotalRunEpisodes/2.)):
                        printNonzeroEntries(l)
                        inputAndEmbed()
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

        print("normfirst {}".format(l.normFirst))
        print("dynafirst {}".format(l.dynaFirst))

        # plt.figure()
        # x = [i for i in range(630)]
        # y = []
        # for key in x:
        #     if key in l.visitedStatesModelCount.keys():
        #         y.append(l.visitedStatesModelCount[key])
        #     else:
        #         y.append(0)

        # plt.bar(x,y)
        # plt.savefig("visitedStatesModelCount-{}-{}".format(descriptionString,i))

        plt.figure()
        plt.plot(o.allmotion)
        plt.savefig("allMotion-{}-{}".format(descriptionString,i))


        allEpisodes.append(allSteps)
        print("All data:")
        for entry in allSteps:
            print("      ", entry)
        print("Final Mean: ", np.mean(allSteps))
        print("Final stdev: ", np.std(allSteps))

        plt.plot(allSteps)
        plt.savefig("experiment-{}-{}".format(descriptionString,i))

        # plot normalized polar to see how we're learning
        # theta = [round(entry, 2) for entry in l.Q]
        # r = [sum(entries) for entries in l.Q]
        # r /= np.max(r)

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='polar')
        # c = ax.scatter(theta, r, cmap='hsv', alpha=0.75)
        # plt.savefig("polarplot{}-exp{}-{}".format(descriptionString,i,numEpisodes))
        # embed()

        # fig = plt.figure()
        # plt.bar(theta, r, linewidth=0.1, edgecolor="black")
        # plt.savefig("barplot{}-exp{}-{}".format(descriptionString,i,numEpisodes))

            # if (save):
        # print("Saving learned values and model")
        # l.save()

        arm.viewer.close()
        del arm
        del l

        # embed()

    allMeans = np.mean([np.mean(entry) for entry in allEpisodes])
    print("allMeans: ", allMeans)

    episodicMean = []
    for i in range(len(allEpisodes[0])):
        total = [entry[i] for entry in allEpisodes]
        episodicMean.append(np.mean(total))

    plt.figure()
    plt.plot(episodicMean)
    plt.savefig("episodicMean-{}-{}".format(descriptionString, i))

    return episodicMean


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

    numExperiments = 1
    TotalRunEpisodes = 100

    # withoutDyna = runAll(TotalRunEpisodes, numExperiments, dynaQ=False)
    withDyna = runAll(TotalRunEpisodes, numExperiments, dynaQ=True)
    
    # plt.figure()
    # x = [i for i in range(len(withDyna))]
    # plt.plot(x, withDyna, x, withoutDyna)
    # plt.legend(['withDyna', 'withoutDyna'])
    # plt.savefig("episodicComparison-{}experiments-{}episodes".format(numExperiments, TotalRunEpisodes))

    # np.save("analytics", np.array(allEpisodes))
    # if (save):
    #     print("Saving learned values and model")
    #     l.save()
    # print("what")
    # # plt.figure(0)
    # print("what1")
    # plt.subplot(111)
    # print("what2")
    # plt.plot([i for i in range(len(allRewards))], allRewards)
    # print("what3")
    # plt.show()
    # print("Training complete.")