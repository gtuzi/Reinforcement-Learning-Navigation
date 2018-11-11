import numpy as np
import random
from collections import namedtuple, deque
from models.model import QNetwork, VizQNet, DuelingQNetwork
from skimage.color import convert_colorspace
import torch
import torch.optim as optim
from torch.autograd import Variable

from buffers.ReplayBuffer import ReplayBuffer
from buffers.PrioritizedReplayBuffer import PrioritizedReplayBuffer
from buffers.MinorityResampledReplayBuffer import MinorityResampledReplayBuffer

import matplotlib.pyplot as plt

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 0.2 # 1e-3        # for soft update of target parameters
LR = 0.001 # 5e-4       # learning rate

REPLAY_EVERY = 2  # For prioritized buffer: how often to learn
UPDATE_EVERY = 20 # 20 # how often to update the target network with the weights of the local network
SEQ_LEN = 10       # Sequence length - for temporal storage of images


class Agent():
    """Interacts with and learns from the environment."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    def __init__(self, state_size, action_size, seed, sample_method = 'uniform', method='doubledqn', device = None, **kwargs):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.device = Agent.device if device is None else device

        # Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(self.device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        self.dqnmethod = method
        self.sample_method = sample_method

        # Replay memory
        if sample_method == 'minority_resampled':
            self.memory = MinorityResampledReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed, device=self.device)
        elif sample_method == 'prioritized':
            if 'alpha' in kwargs:
                alpha = kwargs['alpha']
            else:
                alpha = None
            if 'beta0' in kwargs:
                beta = kwargs['beta0']
            else:
                beta = None
            self.memory = PrioritizedReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed,
                                                  device=self.device, alpha=alpha, beta0=beta)
        elif sample_method == 'uniform':
            self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed, device=self.device)
        else:
            raise Exception('Unrecognized sampling method')

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.update_step = 0
        self.replay_step = 0

        self.episode_count = 0


    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add_new_experience(state, action, reward, next_state, done)

        self.update_step = (self.update_step + 1) % UPDATE_EVERY # This is checked in learn
        self.replay_step = (self.replay_step + 1) % REPLAY_EVERY
        self.episode_count = self.episode_count+1 if done else self.episode_count

        if len(self.memory) > BATCH_SIZE:
            if self.sample_method == 'prioritized':
                if self.replay_step == 0:
                    self.learn(GAMMA, method=self.dqnmethod)
            else:
                self.learn(GAMMA, method=self.dqnmethod)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def dqn(self, rewards, next_states, dones, gamma):
        y = torch.zeros_like(rewards)

        # For end of episode, the return is just the final reward
        y[(dones == 1).squeeze(), ...] = rewards[(dones == 1).squeeze(), ...]

        # Compute the aproximation of the optimal target reward values (Q*)
        with torch.no_grad():
            logits = self.qnetwork_target(next_states[(dones == 0).squeeze(), ...])
            next_values, _ = torch.max(logits, 1)  # Values of next max actions
            y[(dones == 0).squeeze(), ...] = rewards[(dones == 0).squeeze(), ...] + gamma * next_values.unsqueeze(-1)

        return y

    def doubledqn(self, rewards, next_states, dones, gamma):
        y = torch.zeros_like(rewards)

        # For end of episode, the return is just the final reward
        y[(dones == 1).squeeze(), ...] = rewards[(dones == 1).squeeze(), ...]

        # Compute the aproximation of the optimal target reward values (Q*)
        with torch.no_grad():
            # 1 - Get the local net next action
            next_local_logits = self.qnetwork_local(next_states[(dones == 0).squeeze(), ...])
            _, max_next_local_act = torch.max(next_local_logits, 1)  # Values of next max actions

            # 2 - Get target network's value of the local's max next action
            next_target_logits = self.qnetwork_target(next_states[(dones == 0).squeeze(), ...])
            values = next_target_logits.gather(1, max_next_local_act.unsqueeze(-1))

            # 3 - Obtain the approximation of
            y[(dones == 0).squeeze(), ...] = rewards[(dones == 0).squeeze(), ...] + gamma * values

        return y

    def learn(self, gamma, method = 'doubledqn'):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, idc, weights = self.memory.sample(self.episode_count)
        dones = torch.round(dones).int()

        if method == 'dqn':
            y = self.dqn(rewards=rewards, next_states=next_states, dones=dones, gamma=gamma)
        elif method == 'doubledqn':
            y = self.doubledqn(rewards=rewards, next_states=next_states, dones=dones, gamma=gamma)
        else:
            raise Exception('Unrecognized method')

        ## TODO: compute and minimize the loss
        # GT: We train the local network,
        # and update the target network parameters
        # zero the parameter gradients

        self.qnetwork_local.train()

        # 1 - Clear out gradients from the local network: perform detach() and zero_() on network parameters
        self.optimizer.zero_grad()

        # 2 - Local estimation of action values
        local_q = self.qnetwork_local(states)

        # 3 - Loss between the approximation of optimal target reward values (Q*) and local estimates
        local_q = local_q.gather(1, actions) # Prior expected returns

        # Temporal Difference (TD) error
        td_error = y - local_q

        # Update the priorities
        self.memory.update_priorities(np.abs(td_error.data.clone().cpu().numpy()) + 1.0e-5, idc)

        if self.sample_method == 'prioritized':
            local_q.backward(-weights * td_error)
        else:
            loss = torch.nn.MSELoss(reduce=False)(local_q, y)
            # 4 - Gradient descend on local network
            loss.backward(weights)

        # 5 - Gradient update
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        if self.update_step == 0:
            # 6 - Set local network to eval
            # self.qnetwork_local.eval() ... this messes things up !!
            self.soft_update(TAU)

    def soft_update(self, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class DuelingAgent():
    """Interacts with and learns from the environment."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __init__(self, state_size, action_size, seed, sample_method='uniform', method='doubledqn', device=None,
                 **kwargs):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.device = Agent.device if device is None else device

        # Network
        self.qnetwork_local = DuelingQNetwork(state_size, action_size, seed).to(self.device)
        self.qnetwork_target = DuelingQNetwork(state_size, action_size, seed).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        self.dqnmethod = method
        self.sample_method = sample_method

        # Replay memory
        if sample_method == 'minority_resampled':
            self.memory = MinorityResampledReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed, device=self.device)
        elif sample_method == 'prioritized':
            if 'alpha' in kwargs:
                alpha = kwargs['alpha']
            else:
                alpha = None
            if 'beta0' in kwargs:
                beta = kwargs['beta0']
            else:
                beta = None
            self.memory = PrioritizedReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed,
                                                  device=self.device, alpha=alpha, beta0=beta)
        elif sample_method == 'uniform':
            self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed, device=self.device)
        else:
            raise Exception('Unrecognized sampling method')

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.update_step = 0
        self.replay_step = 0

        self.episode_count = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add_new_experience(state, action, reward, next_state, done)

        self.update_step = (self.update_step + 1) % UPDATE_EVERY  # This is checked in learn
        self.replay_step = (self.replay_step + 1) % REPLAY_EVERY
        self.episode_count = self.episode_count + 1 if done else self.episode_count

        if len(self.memory) > BATCH_SIZE:
            if self.sample_method == 'prioritized':
                if self.replay_step == 0:
                    self.learn(GAMMA, method=self.dqnmethod)
            else:
                self.learn(GAMMA, method=self.dqnmethod)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def dqn(self, rewards, next_states, dones, gamma):
        y = torch.zeros_like(rewards)

        # For end of episode, the return is just the final reward
        y[(dones == 1).squeeze(), ...] = rewards[(dones == 1).squeeze(), ...]

        # Compute the aproximation of the optimal target reward values (Q*)
        with torch.no_grad():
            logits = self.qnetwork_target(next_states[(dones == 0).squeeze(), ...])
            next_values, _ = torch.max(logits, 1)  # Values of next max actions
            y[(dones == 0).squeeze(), ...] = rewards[(dones == 0).squeeze(), ...] + gamma * next_values.unsqueeze(-1)

        return y

    def doubledqn(self, rewards, next_states, dones, gamma):
        y = torch.zeros_like(rewards)

        # For end of episode, the return is just the final reward
        y[(dones == 1).squeeze(), ...] = rewards[(dones == 1).squeeze(), ...]

        # Compute the aproximation of the optimal target reward values (Q*)
        with torch.no_grad():
            # 1 - Get the local net next action
            next_local_logits = self.qnetwork_local(next_states[(dones == 0).squeeze(), ...])
            _, max_next_local_act = torch.max(next_local_logits, 1)  # Values of next max actions

            # 2 - Get target network's value of the local's max next action
            next_target_logits = self.qnetwork_target(next_states[(dones == 0).squeeze(), ...])
            values = next_target_logits.gather(1, max_next_local_act.unsqueeze(-1))

            # 3 - Obtain the approximation of
            y[(dones == 0).squeeze(), ...] = rewards[(dones == 0).squeeze(), ...] + gamma * values

        return y

    def learn(self, gamma, method='doubledqn'):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, idc, weights = self.memory.sample(self.episode_count)
        dones = torch.round(dones).int()

        if method == 'dqn':
            y = self.dqn(rewards=rewards, next_states=next_states, dones=dones, gamma=gamma)
        elif method == 'doubledqn':
            y = self.doubledqn(rewards=rewards, next_states=next_states, dones=dones, gamma=gamma)
        else:
            raise Exception('Unrecognized method')

        ## TODO: compute and minimize the loss
        # GT: We train the local network,
        # and update the target network parameters
        # zero the parameter gradients

        self.qnetwork_local.train()

        # 1 - Clear out gradients from the local network: perform detach() and zero_() on network parameters
        self.optimizer.zero_grad()

        # 2 - Local estimation of action values
        local_q = self.qnetwork_local(states)

        # 3 - Loss between the approximation of optimal target reward values (Q*) and local estimates
        local_q = local_q.gather(1, actions)  # Prior expected returns

        # Temporal Difference (TD) error
        td_error = y - local_q

        # Update the priorities
        self.memory.update_priorities(np.abs(td_error.data.clone().cpu().numpy()) + 1.0e-5, idc)

        if self.sample_method == 'prioritized':
            local_q.backward(-weights * td_error)
        else:
            loss = torch.nn.MSELoss(reduce=False)(local_q, y)
            # 4 - Gradient descend on local network
            loss.backward(weights)

        # 5 - Gradient update
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        if self.update_step == 0:
            # 6 - Set local network to eval
            # self.qnetwork_local.eval() ... this messes things up !!
            self.soft_update(TAU)

    def soft_update(self, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class VisualAgent():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __init__(self, state_size, action_size, seed, sample_method='uniform', method='dqn', device = None, **kwargs):
        """Initialize an Agent object.

        Params
        ======
            state_size (c:int x h:int x w:int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.seqlen = SEQ_LEN
        self.device = VisualAgent.device if device is None else device

        # Q-Network
        self.qnetwork_local = VizQNet(self.seqlen, action_size, seed).to(self.device)
        self.qnetwork_target = VizQNet(self.seqlen, action_size, seed).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        self.dqnmethod = method
        self.sample_method = sample_method
        # Replay memory
        if sample_method == 'minority_resampled':
            self.memory = MinorityResampledReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed, device=self.device)
        elif sample_method == 'prioritized':
            if 'alpha' in kwargs:
                alpha = kwargs['alpha']
            else:
                alpha = None
            if 'beta0' in kwargs:
                beta = kwargs['beta0']
            else:
                beta = None
            self.memory = PrioritizedReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed,
                                                  device=self.device, alpha=alpha, beta0=beta)
        elif sample_method == 'uniform':
            self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed, device=self.device)
        else:
            raise Exception('Unrecognized sampling method')

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.update_step = 0
        self.replay_step = 0
        self.episode_count = 0


    def step(self, x, action, reward, next_x, done):

        # Generate the state
        state = np.array(self.sequence)[None, ...]

        # Append the new image, and create the next state
        self._preprocess_(next_x)
        next_state = np.array(self.sequence)[None, ...]

        # Save experience in replay memory
        self.memory.add_new_experience(state, action, reward, next_state, done)

        self.update_step = (self.update_step + 1) % UPDATE_EVERY # This is checked in learn
        self.replay_step = (self.replay_step + 1) % REPLAY_EVERY
        self.episode_count = self.episode_count + 1 if done else self.episode_count

        if len(self.memory) > BATCH_SIZE:
            if (self.sample_method == 'prioritized'):
                if (self.replay_step == 0):
                    self.learn(GAMMA, method=self.dqnmethod)
            else:
                self.learn(GAMMA, method=self.dqnmethod)


    def on_new_episode(self, x1):
        self.sequence = deque(maxlen=self.seqlen) # Reset the sequence
        # On new episode, just repeat the first image in the sequence
        for _ in range(self.seqlen):
            self._preprocess_(np.copy(x1))


    def _preprocess_(self, xi):
        '''
            Pre-process the sequence to state.
            We're using a deque, which inserts left to right. This pre-processes
            only the most recent image
        :return:
        '''

        xi = xi.squeeze()
        # Un-normalize
        xi = np.round((xi*255.)).astype(np.uint8)
        x = convert_colorspace(xi, fromspace='rgb', tospace='YCbCr')[..., 0] # Get only the chroma
        self.sequence.append(x)


    def act(self, x, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current image
            eps (float): epsilon, for epsilon-greedy action selection
        """
        # Generate the state
        state = np.array(self.sequence)[None, ...]
        state = torch.from_numpy(state).float().to(self.device)

        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))


    def dqn(self, rewards, next_states, dones, gamma):
        y = torch.zeros_like(rewards)

        # For end of episode, the return is just the final reward
        y[(dones == 1).squeeze(), ...] = rewards[(dones == 1).squeeze(), ...]

        # Compute the aproximation of the optimal target reward values (Q*)
        with torch.no_grad():
            logits = self.qnetwork_target(next_states[(dones == 0).squeeze(), ...])
            next_values, _ = torch.max(logits, 1)  # Values of next max actions
            y[(dones == 0).squeeze(), ...] = rewards[(dones == 0).squeeze(), ...] + gamma * next_values.unsqueeze(-1)

        return y


    def doubledqn(self, rewards, next_states, dones, gamma):
        y = torch.zeros_like(rewards)

        # For end of episode, the return is just the final reward
        y[(dones == 1).squeeze(), ...] = rewards[(dones == 1).squeeze(), ...]

        # Compute the aproximation of the optimal target reward values (Q*)
        with torch.no_grad():
            # 1 - Get the local net next action
            next_local_logits = self.qnetwork_local(next_states[(dones == 0).squeeze(), ...])
            _, max_next_local_act = torch.max(next_local_logits, 1)  # Values of next max actions

            # 2 - Get target network's value of the local's max next action
            next_target_logits = self.qnetwork_target(next_states[(dones == 0).squeeze(), ...])
            values = next_target_logits.gather(1, max_next_local_act.unsqueeze(-1))

            # 3 - Obtain the approximation of
            y[(dones == 0).squeeze(), ...] = rewards[(dones == 0).squeeze(), ...] + gamma * values

        return y


    def learn(self, gamma, method='doubledqn'):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, idc, weights = self.memory.sample(self.episode_count)
        dones = torch.round(dones).int()

        if method == 'dqn':
            y = self.dqn(rewards=rewards, next_states=next_states, dones=dones, gamma=gamma)
        elif method == 'doubledqn':
            y = self.doubledqn(rewards=rewards, next_states=next_states, dones=dones, gamma=gamma)
        else:
            raise Exception('Unrecognized method')

        ## TODO: compute and minimize the loss
        # GT: We train the local network,
        # and update the target network parameters
        # zero the parameter gradients

        self.qnetwork_local.train()

        # 1 - Clear out gradients from the local network
        self.optimizer.zero_grad()

        # 2 - Local estimation of action values
        local_q = self.qnetwork_local(states)

        # 3 - Loss between the approximation of optimal target reward values (Q*) and local estimates
        local_q = local_q.gather(1, actions)  # Prior expected returns

        # Temporal Difference (TD) error
        td_error = y - local_q

        # Update the priorities
        self.memory.update_priorities(np.abs(td_error.data.clone().cpu().numpy()) + 1.0e-5, idc)

        if self.sample_method == 'prioritized':
            local_q.backward(-weights * td_error)
        else:
            loss = torch.nn.MSELoss(reduce=False)(local_q, y)
            # 4 - Gradient descend on local network
            loss.backward(weights)

        # 5 - Gradient update
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        if self.update_step == 0:
            # 6 - Set local network to eval
            # self.qnetwork_local.eval()
            self.soft_update(TAU)

    def soft_update(self, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)





