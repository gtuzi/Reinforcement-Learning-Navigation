import numpy as np
import random
from recordclass import recordclass
import torch
import math
from collections import deque

class PrioritizedReplayBuffer:
    """Fixed-size buffer to store experience tuples."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __init__(self, action_size, buffer_size, batch_size, seed, alpha = None, beta0 = None, device = None):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """

        self.action_size = action_size
        self.buffer_size = buffer_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = recordclass("Experience", field_names=["state", "action", "reward", "next_state", "done", "priority"])
        self.seed = random.seed(seed)
        self.max_priority = 1.
        self.alpha = 1. if alpha is None else alpha
        self.beta0 = 1. if beta0 is None else beta0
        self.device = PrioritizedReplayBuffer.device if device is None else device
        pass

    def add_new_experience(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""

        e = self.experience(state, action, reward, next_state, done, self.max_priority)
        self.memory.append(e)

    def update_priorities(self, tde, idc):
        '''
            Update the priorities of experiences with the evaluated TD error
        :param tde: TD error
        :param idc: indeces of each of the samples, corresponding to the indeces of the samples in the experience
        :return:
        '''

        assert np.sum(np.isnan(tde)) == 0
        assert np.sum(np.isinf(tde)) == 0
        for i, ix in enumerate(idc):
            self.memory[ix].priority = float(tde[i, 0])
        self._update_max_priority()

    def _update_max_priority(self):
        '''
            Update the value of the maximal priority.
        :return:
        '''
        self.max_priority = -math.inf
        for experience in self.memory:
            if self.max_priority < experience.priority:
                self.max_priority = experience.priority

        assert self.max_priority != (-math.inf)

    @staticmethod
    def _softmax_a(x, alpha):
        x = x**alpha
        e_x = np.exp(x - np.max(x))
        return e_x/e_x.sum(axis=0)

    def sample(self, episode_count):
        """Randomly sample a batch of experiences from memory."""
        # Annealing the coefficients
        # self.alpha = min(1., 1.0001*self.alpha)
        beta = 1. if episode_count >= 2000 else np.linspace(self.beta0, 1., 2000)[episode_count]

        # Rebalance priorities. Keep track of the experience buffer indeces
        priorities = np.array([(i, experience.priority) for i, experience in enumerate(self.memory) if experience is not None])
        idc, priorities = np.round(priorities[..., 0]).astype(np.int), priorities[..., 1].astype(np.float32)

        # Paper method
        # priorities = (priorities**self.alpha) / np.sum(priorities**self.alpha + 1.0e-12) # Avoid division by 0
        # Adjust to add up to 1 - due to float32 roundoff error
        # priorities += (1. - np.sum(priorities)) / float(priorities.shape[0])

        # Using softmax
        priorities = PrioritizedReplayBuffer._softmax_a(priorities, self.alpha)
        priorities[priorities < 1.0e-12] = 0.

        # Priority sample: sample the indeces of the memory tuples according to the priorities
        iii = np.random.choice(list(range(idc.shape[0])), size=self.batch_size, p=priorities, replace=True)
        idc = idc[iii, ...].tolist()
        priorities = priorities[iii, ...]
        assert np.sum(np.isnan(priorities)) == 0

        # Compute the IS weights
        weights = ((self.buffer_size*priorities) ** (-beta))[..., None].astype(np.float32)
        # Scale the weights
        weights /=np.max(weights)
        assert np.sum(np.isnan(weights)) == 0

        # Collect the experiences we'll be using
        # No need to check for None experience here since we checked above
        experiences = [self.memory[i] for i in idc]

        # Transform into pytorch
        states = torch.from_numpy(np.vstack([e.state for e in experiences])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float().to(self.device)
        weights = torch.from_numpy(weights).float().to(self.device)
        return states, actions, rewards, next_states, dones, idc, weights

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
