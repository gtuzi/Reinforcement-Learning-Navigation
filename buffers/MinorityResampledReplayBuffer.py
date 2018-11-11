import numpy as np
import random
from collections import deque
from recordclass import recordclass
import torch
from imblearn.over_sampling import RandomOverSampler
from sklearn.utils import resample
from scipy.stats import iqr

class MinorityResampledReplayBuffer:
    """Fixed-size buffer to store experience tuples."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __init__(self, action_size, buffer_size, batch_size, seed, device = None):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = recordclass("Experience", field_names=["state", "action", "reward", "next_state", "done", "priority"])
        self.seed = random.seed(seed)
        self.max_priority = 1.
        self.device = MinorityResampledReplayBuffer.device if device is None else device

    def add_new_experience(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done, self.max_priority)
        self.memory.append(e)

    def update_priorities(self, td_error, idc):
        for i, ix in enumerate(list(set(idc))):
            self.memory[ix].priority = float(td_error[i, 0])
            # Capture maximal priority
            if self.max_priority < float(td_error[i, 0]):
                self.max_priority = float(td_error[i, 0])


    def sample(self, episode_count = None):
        """Randomly sample a batch of experiences from memory."""

        # Keep track of the experience buffer indeces
        priorities = np.array([(i, experience.priority) for i, experience in enumerate(self.memory) if experience is not None])
        idc, priorities = np.round(priorities[..., 0]).astype(np.int), priorities[..., 1]

        # Oversample minority classes
        resamp_idc = MinorityResampledReplayBuffer.minority_oversample(idc, priorities, ratio='all', post_resamp_n = self.batch_size).squeeze()
        experiences = [self.memory[i] for i in list(resamp_idc) if self.memory[i] is not None]
        weights = torch.from_numpy(np.ones(shape=(self.batch_size, 1), dtype=np.float32))
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            self.device)

        return states, actions, rewards, next_states, dones, list(resamp_idc), weights

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

    @staticmethod
    def minority_oversample(x, labels, ratio='auto', post_resamp_n=None):
        """
        Over-sample the minority class(es) in "labels" by picking samples at random.
        with replacement.
        :param x: data to resample
        :param labels: values of dependent variable (m - samples,)
        :param binwidth: width of bin by which to group values. Bins will be the discrete "label" by which to randomly over-sample
        :param post_resamp_n: after resampling, unformly subsample - without replacement - this many samples.
        :return:
        resampled x
        """

        binwidth = 2*(iqr(labels))/(labels.shape[0]**(1./3))

        miny, maxy = np.min(labels), np.max(labels)
        # Center the min/maxes inside their own bins
        yrange = maxy - miny
        nbins = np.ceil(yrange / (binwidth + 1.0e-10))

        if nbins == 0:
            return x

        # Obtain bin centers, where min/max are within the beginning/ending bins
        binsctrs = np.linspace((miny - binwidth / 2000.), (maxy + binwidth / 2000.), num=nbins)

        # Assign y-values to each bin
        bins = np.digitize(x=labels, bins=binsctrs)

        # Use the index of each sample to over-sample.
        # 1) Pair each sample with an index.
        # 2) Resample the p
        # Represent the resample array as: [y-value, sample-index]
        samp_idc_feat = np.array(list(range(labels.shape[0])))[..., None]

        # Let n_maj be the number of the majority class. ROS generates data
        # such that the count for each class is equal to n_maj.
        ros = RandomOverSampler(ratio)
        resamp_idc, res_bins = ros.fit_sample(X=samp_idc_feat, y=bins)

        if post_resamp_n is not None:
            assert post_resamp_n <= resamp_idc.shape[0]
            resamp_idc, _ = resample(resamp_idc, res_bins, n_samples=post_resamp_n, replace=False)


        # Recover the y-values from each sample
        return x[resamp_idc, ...]