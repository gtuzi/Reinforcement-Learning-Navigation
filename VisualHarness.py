from unityagents import UnityEnvironment
import numpy as np
from collections import deque
import torch
import matplotlib.pyplot as plt
from skimage.color import convert_colorspace
from skimage import exposure

from agents.dqn_agent import VisualAgent


env = UnityEnvironment(file_name="VisualBanana.app")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents in the environment
print('Number of agents:', len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space
state = env_info.visual_observations[0]

print('States look like:')
plt.imshow(np.squeeze(state))
plt.show()
state_size = state.shape
print('States have shape:', state.shape)

xi = np.round((np.squeeze(state)*255.)).astype(np.uint8)
x = convert_colorspace(xi, fromspace='rgb', tospace='YCbCr')[..., 0] # Get only the Luminance
plt.imshow(np.squeeze(x))
plt.show()


def pixel_dqn(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995, prefix='dqn'):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]           # reset the environment
        image = env_info.visual_observations[0]                     # get the current image
        score = 0                                                   # initialize the score

        agent.on_new_episode(image)

        for t in range(max_t):
            action = agent.act(image, eps)                          # select an action
            env_info =  env.step(action)[brain_name]                # send the action to the environment
            next_image = env_info.visual_observations[0]            # get the next image
            reward = env_info.rewards[0]                            # get the reward
            done = env_info.local_done[0]                           # see if episode has finished
            agent.step(state, action, reward, next_image, done)
            image = next_image
            score += reward
            if done:
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon

        print('\rEpisode {}\tAverage Score: {:.2f}, Recent Score: {:.2f}'.format(i_episode, np.mean(scores_window), score), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= 13.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f},  Recent Score: {:.2f}'.format(
                i_episode - 100,
                np.mean(scores_window), score))
            torch.save(agent.qnetwork_local.state_dict(), '{}_checkpoint.pth'.format(prefix))
            break

    return scores



agent = VisualAgent(state_size=(state_size[-1], state_size[1], state_size[2]),
                    action_size=action_size,
                    sample_method='uniform',
                    seed=13)
scores = pixel_dqn(eps_start=0.3)


# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
