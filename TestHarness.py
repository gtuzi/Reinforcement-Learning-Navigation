from unityagents import UnityEnvironment
import numpy as np
from collections import deque
import torch
import matplotlib.pyplot as plt
import time
from agents.dqn_agent import Agent

env = UnityEnvironment(file_name="Banana.app", seed=13)
time.sleep(5.)
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
state = env_info.vector_observations[0]
print('States look like:', state)
state_size = len(state)
print('States have length:', state_size)


method = 'doubledqn'
alpha = 1.
beta0 = 0.0
agent = Agent(state_size=state_size, action_size=action_size,
              sample_method = 'prioritized',
              method=method, seed=0, alpha = alpha, beta0 = beta0)

# Load model
# epochs = list(range(100, 3100, 100))
# epochs = [100, 200, 300, 400, 1000, 1500, 2000, 2500, 3000]
epochs=  [2900]
eps = 0.0
for epoch_idx in epochs:
    prefix = 'Unif_ddqn_' + method
    print('Epoch: {}'.format(epoch_idx))

    agent.qnetwork_local.load_state_dict(torch.load('./models_deleteme/{}_checkpoint_{}.pth'.format(prefix, epoch_idx)))
    env_info = env.reset(train_mode=False)[brain_name]
    state = env_info.vector_observations[0].astype(np.float32)  # get the current state

    while True:
        action = agent.act(state, eps)
        env_info = env.step(action)[brain_name]
        state = env_info.vector_observations[0].astype(np.float32)
        done = env_info.local_done[0]
        if done:
            print('Game over !!!')
            break
        else:
            # time.sleep(1./66)
            pass

env.close()