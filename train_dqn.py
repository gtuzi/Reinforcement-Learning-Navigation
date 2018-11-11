from unityagents import UnityEnvironment
import numpy as np
import random


def _get_env_():
    # Run the appropriate simulator according to OS
    from sys import platform as _platform
    if _platform == "linux" or _platform == "linux2":
        # linux
        env = UnityEnvironment(file_name="./Banana_Linux/Banana.x86_64")
    elif _platform == "darwin":
        # MAC OS X
        env = UnityEnvironment(file_name="Banana.app")
    else:
        raise Exception('Use of Windows is punishable by herd shame')
    return env


def start_env():
    env = _get_env_()
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    # reset the environment
    env_info = env.reset(train_mode=False)[brain_name]
    # number of actions
    action_size = brain.vector_action_space_size
    # examine the state space
    state = env_info.vector_observations[0]
    state_size = len(state)
    return env, state_size, action_size, brain_name

#####
from collections import deque
import torch
import matplotlib.pyplot as plt
from agents.dqn_agent import Agent

def dqn(env, agent,
        n_episodes=2000,
        max_t=1000,
        eps_start=1.0,
        eps_end=0.01, eps_decay=0.995,
        walk_penalty=0., prefix='dqn'):
    """
        Training DQN (or DDQN)
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
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]  # get the current state
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]

            eps = 1.0e-2
            if (reward < eps) and (reward > -eps):
                reward = walk_penalty

            done = env_info.local_done[0]
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon

        print('\rEpisode {}\tAverage Score: {:.2f}, Recent Score: {:.2f}'.format(i_episode, np.mean(scores_window),
                                                                                 score), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= 13.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f},  Recent Score: {:.2f}'.format(
                i_episode - 100,
                np.mean(scores_window), score))
            save_path = './models/{}_checkpoint.pth'.format(prefix)
            torch.save(agent.qnetwork_local.state_dict(), save_path)
            print('Model saved as: {}'.format(save_path))
            break

    return scores



import sys, getopt
if __name__== "__main__":
    # Defaults
    method = 'doubledqn'
    sample_method = 'uniform'
    alpha = 0.6
    beta0 = 0.4
    walk_penalty = 0.0
    max_t = 1000
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Parse arguments and options
    try:
        opts, args = getopt.getopt(sys.argv[1:], shortopts = "m:s:a:b:w:d:t:")
    except getopt.GetoptError as ex:
        print( 'train_dqn.py -m <dqn or doubledqn(default)> -s <prioritized or uniform (default)> ' +\
               '-a <value (0.6 default)> -b <value (0.4 default)> -w <value, (0.0 default)> '+
               '-t <value (1000 default)> -d <gpu or cpu (defaults to gpu if available)>')
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-m",):
            method = str(arg)
        elif opt in ("-s", ):
            sample_method = str(arg)
        elif opt in ("-a", ):
            alpha = float(arg)
        elif opt in ("-b", ):
            beta0 = float(arg)
        elif opt in ("-w", ):
            walk_penalty = float(arg)
        elif opt in ("-d", ):
            device = str(arg)
        elif opt in ("-t", ):
            max_t = int(max_t)



    env, state_size, action_size, brain_name = start_env()
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    agent = Agent(state_size=state_size,
                  action_size=action_size,
                  walk_penalty=walk_penalty,
                  sample_method=sample_method,
                  method=method,
                  device=device,
                  alpha=alpha, beta0=beta0,
                  seed=random.randint(0, 99999))

    scores = dqn(env, agent=agent, eps_start=1.0, eps_decay=0.995, prefix=sample_method + '_' + method)





