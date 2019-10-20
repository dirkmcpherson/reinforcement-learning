This programming assignment focuses on excercise 8.4 of Sutton & Barto. It implements Dyna-Q+ to explore a maze while learning from the model in the background. 

$\epsilon-greedy$ is not sufficient to explore long paths through the environment that our model has deemed as non-ideal. If our model is wrong, or if the environment is non-stationary, we may never explore optimal paths. Dyna-Q+ aims to solve this by tracking how long its been since a state-action pair have been tried in the real environment. The idea is to check out states that have not been explored in awhile to give the agent the opportunity to update its model.

While Dyna-Q+ is planning in the background, it adds $\k*sqrt(\tau)$ to a transitions reward. Where $\k$ is some small constant and $\tau$ is the number of timesteps since the transition was taken. The apparent reward of less explored transitions grows until, occasionally the agent explores whole paths it has previously considered useless. 

