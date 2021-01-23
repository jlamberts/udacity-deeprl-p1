import torch.optim as optim
import torch
import random
import numpy as np
import torch.nn.functional as F

from collections import namedtuple, deque

from navigation.networks import QNetwork

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent:
    """Interacts with and learns from the environment."""

    def __init__(
        self,
        state_size,
        action_size,
        seed=None,
        double_dqn=False,
        network=QNetwork,
        network_kwargs=None,
        buffer_size=int(1e5),
        batch_size=64,
        gamma=0.99,
        tau=1e-3,
        lr=5e-4,
        update_every=4,
    ):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            double_dqn (bool): if True, use a double DQN implementation; otherwise, use standard DQN.
                In standard DQN, the same network both chooses the action that will be taken in state t+1 and evaluates the
                expected reward for that action; in double DQN, the "local" network chooses the action, while the "target"
                network evaluates it.
            network: (nn.Module): Network class that accepts states as inputs and outputs Q-values for each action
            buffer_size: (int): Replay buffer size
            batch_size: (int): Minibatch size
            gamma: (float): Discount factor
            tau: (float): Soft update rate for target network
            lr: (float): Learning rate
            update_every: (int): How often to update the network
        """
        self.state_size = state_size
        self.action_size = action_size
        if seed:
            self.seed = random.seed(seed)
        self.double_dqn = double_dqn
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.update_every = update_every

        # Q-Network
        network_kwargs = network_kwargs if network_kwargs else {}
        self.network_class = network
        if seed:
            self.qnetwork_local = network(state_size=state_size, action_size=action_size, seed=seed, **network_kwargs).to(device)
            self.qnetwork_target = network(state_size=state_size, action_size=action_size, seed=seed, **network_kwargs).to(device)
        else:
            self.qnetwork_local = network(state_size=state_size, action_size=action_size, **network_kwargs).to(device)
            self.qnetwork_target = network(state_size=state_size, action_size=action_size, **network_kwargs).to(device)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)

        # Replay memory
        self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)

    def act(self, state, eps=0.0):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # compute target
        if self.double_dqn:
            next_actions = self.qnetwork_local(next_states).argmax(dim=1, keepdim=True)
            next_q_value = self.qnetwork_target(next_states).gather(1, next_actions)
        else:
            next_q_value, _ = self.qnetwork_target(next_states).max(dim=1, keepdim=True)
        y_i = rewards + next_q_value * gamma

        # compute current
        current_q_value = self.qnetwork_local(states).gather(1, actions)

        # calculate loss
        loss = F.mse_loss(current_q_value, y_i)

        # update local params
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.qnetwork_local.parameters():
            # for stability
            param.grad.clamp(-1, 1)
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
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
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
