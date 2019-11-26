import numpy as np
import learner
import subjective
import env
import action

def main():
    pass

def reset(arm):
    arm.reset()


if __name__ == '__main__':
    numEpisodes = 100
    action_resolution = 1 # how many timesteps between updates?
    step_count = 0

    arm = env.ArmEnv()
    o = subjective.Subjective(arm)
    l = learner.DynaQLearner()
    actionIdx = action.getRandomActionIdx()

    indexCount = [0 for entry in action.ActionSet]
    while(True):
        arm.render() # render the arm visually

        TakeStep = (step_count % action_resolution) == 0
        if TakeStep:
            state = arm.get_state()
            actionIdx = l.sampleAction(state)
            # print("Chose action {} from state {}".format(action.ActionSet[actionIdx], state))
            indexCount[actionIdx] += 1

        newState, reward, goalAchieved = arm.step(action.ActionSet[actionIdx]) # select an action based on the current policy

        if (goalAchieved or TakeStep):
            l.update(state, actionIdx, newState, reward)

        # arm.step(arm.sample_action()) # select an action based on the current policy
        feedback = o.evaluateEnvironment(arm) # Oracle observes the environment so check if it has feedback
        if feedback:
            l.acceptFeedback(state)

        if (goalAchieved):
            print("Episode complete.")
            reset(arm)

            numEpisodes -= 1
            if (numEpisodes <= 0):
                break
        
        step_count += 1

    print("Training complete.")