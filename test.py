# Tennis Agent Testing

# Import libraries
from unityagents import UnityEnvironment
import numpy as np
import torch

# Create instance of the Tennis environment
env = UnityEnvironment(file_name='Tennis.app') # Update the app name/location if not using macOS

# Get brain

# Get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# Load Actor model weights for both agents
from model import Actor

agent1 = Actor(state_size=24, action_size=2, seed=0)
agent1.load_state_dict(torch.load('checkpoint_1_actor.pth'))
agent2 = Actor(state_size=24, action_size=2, seed=0)
agent2.load_state_dict(torch.load('checkpoint_2_actor.pth'))

# Testing
def test(state, agent):

    """
    Testing the Reacher agent for a single agent
    Params
    ======
        state (numpy.ndarray): Current state that the agents are experiencing
        agents (int):          The number of agents (= 20 in this case)
        action_size (int):     Number of possible actions an agent can take
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device == "cuda:0":
        agent = agent.cuda()

    state = torch.from_numpy(state).float().to(device)
    agent.eval()
    with torch.no_grad():
        action = agent(state).cpu().data.numpy()

    return np.clip(action, -1, 1)

for i in range(1, 2):                                     # play game for 5 episodes
    env_info = env.reset(train_mode=False)[brain_name]     # reset the environment
    states = env_info.vector_observations                  # get the current state (for each agent)
    scores = np.zeros(2)                          # initialize the score (for each agent)
    while True:
        action1 = test(states[0], agent1)
        action2 = test(states[1], agent2)
        actions = [action1, action2]
        env_info = env.step(actions)[brain_name]           # send all actions to tne environment
        next_states = env_info.vector_observations         # get next state (for each agent)
        rewards = env_info.rewards                         # get reward (for each agent)
        dones = env_info.local_done                        # see if episode finished
        scores += env_info.rewards                         # update the score (for each agent)
        states = next_states                               # roll over states to next time step
        if np.any(dones):                                  # exit loop if episode finished
            break
    print('Score (max over agents) from episode {}: {}'.format(i, np.max(scores)))
env.close()
