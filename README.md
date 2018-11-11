[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"
### Introduction

In this project I trained an agent to navigate and collect as many healthy *yellow* bananas in a large, square world. A world designed to dupe the untrained into also collecting poisonous *blue* bananas.

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  It is the goal of the agent to collect as many healthy bananas while avoiding those poisonous blue ones.  

The sensors (i.e. states) provide 37 measurements, or dimensions. They contain the agent's velocity, along with ray-based perception of objects around agent's forward direction.  
Equipped with this information, the agent has to learn how to best select actions.  *Four* discrete actions are available:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The learning task is *episodic*. The environment is solved when the agent, through its unrelenting perseverance and our tender algorithmic care, achieves an average score of +13 over 100 consecutive episodes.

### Getting Started: The Environment
Environments for downloading:
1.
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
   
2. The following versions were used for this project:
- `python 3.5.5`
- `pytorch 0.4.1`
- `numpy 1.15.2`

### Train agent
The agent is a neural network trained using Deep Q-Learning (DQN). Use the `train_dqn` with the following arguments:
- `-m` training method: `dqn` or `doubledqn`
- `s` experience replay method: `prioritized` or `uniform` sampling of experience
- `a`, `b` prioritized replay hyperparameters. Valid only if `-s prioritized`
- `w`: walk penalty: a non-negative penalization value to add to each bananaless step.
- `t`: maximum time: episode max runtime


### Run agent
From command line, call `python run_dqn -f <model>` where `<model>` is the local location of the model

A few pre-trained models are located under the `models` folder in this repository. Naming convention: 
- `pri` an agent trained with prioritized experience replay [paper](https://arxiv.org/pdf/1511.05952.pdf). The `a` and `b` denote the alpha and beta hyperparameters used to control the importance sampling probability (alpha) and the correction of the bias (beta) introduced by this sampling.   
- `unif` an agent trained with uniform experience replay as used in the Deep Q-Learning [paper](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf)
- `dqn` implements a networks similar to  [paper](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf) while `doubledqn` refers to an agent trained with double Q-Learning as *Hasselt et. al* showed in this [paper](https://arxiv.org/pdf/1509.06461.pdf). 

An example video of an agent is located in `resources/RL_Navigations_Bananas copy.mp4` 


### Experimental
#### Sampling
Minority-resampled experience replay has also been implemented in `MinorityResampledBuffer.py`. TD-error over the memory is first binned, and minority oversampled. This is an experimental implementation and runs quite slow at this point. `imbalanced-learn` v. 0.4.3 package is required.
#### Cost
A "walking" cost can be added during training (option `-w` ). This will add the value supplied to the `0` returns of the original environment. This drives the agent to learn shorter paths to the collection of bananas. I noticed that this option speeds up initial learning greatly.

  