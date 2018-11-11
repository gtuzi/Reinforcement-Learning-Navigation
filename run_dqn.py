from unityagents import UnityEnvironment
import numpy as np
from collections import deque
import torch
import matplotlib.pyplot as plt
import time
from agents.dqn_agent import Agent
from models.model import QNetwork

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


import sys, getopt,os, random
if __name__== "__main__":

    try:
        opts, args = getopt.getopt(sys.argv[1:], "m:")
    except getopt.GetoptError:
        print('Usage: run_dqn.py -m <model file location>')
        sys.exit(2)

    ws = os.getcwd()
    for opt, arg in opts:
        if opt in ("-m",):
            file = str(arg)
            fComplete = ws + '/' + file
            if not os.path.isfile(fComplete):
                print('{} is not found'.format(file))
                sys.exit(2)
        else:
            print('Usage: run_dqn.py -f <model file location>')
            sys.exit(2)

    env, state_size, action_size, brain_name = start_env()

    qnetwork = QNetwork(state_size, action_size, random.randint(0, 99999)).to('cpu')
    method = 'doubledqn'
    alpha = 1.
    beta0 = 0.0

    try:
        qnetwork.load_state_dict(torch.load(fComplete, map_location='cpu'))
        qnetwork.eval()
    except Exception as ex:
        print('Could not load model: {}'.format(str(ex)))
        sys.exit(2)

    env_info = env.reset(train_mode=False)[brain_name]
    state = env_info.vector_observations[0].astype(np.float32)  # get the current state

    try:
        while True:
            with torch.no_grad():
                state_torch = torch.from_numpy(state).float().unsqueeze(0)
                action_values = qnetwork(state_torch)
                action = np.argmax(action_values.cpu().data.numpy())
            env_info = env.step(action)[brain_name]
            state = env_info.vector_observations[0].astype(np.float32)
            done = env_info.local_done[0]
            if done:
                print('Game Over !')
                break
            else:
                # time.sleep(1./66.)
                pass
    finally:
        env.close()