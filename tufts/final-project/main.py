import numpy as np
import learner
import subjective
import env
import action

def main():
    pass

def reset():
    pass


if __name__ == '__main__':
    numEpisodes = 100

    arm = env.ArmEnv()
    o = subjective.Subjective(arm)
    l = learner.DynaQLearner()
    while(True):
        arm.render() # render the arm visually
        state = arm.get_state()
        actionIdx = l.sampleAction(state)
        newState, reward, goalAchieved = arm.step(action.ActionSet[actionIdx]) # select an action based on the current policy
        l.update(state, actionIdx, newState, reward)
        # arm.step(arm.sample_action()) # select an action based on the current policy
        feedback = o.evaluateEnvironment(arm) # Oracle observes the environment so check if it has feedback
        if feedback:
            l.acceptFeedback(state)

        if (goalAchieved):
            print("Episode complete.")
            reset()

            numEpisodes -= 1
            if (numEpisodes <= 0):
                break

    print("Training complete.")