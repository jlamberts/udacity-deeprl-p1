import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size=37, action_size=4, hidden_layer_size=512, seed=42):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        # no need for convolutions like in the paper since we have a low dimensional numeric input space
        # paper architecture is input layer -> 512 hidden units with ReLU activation -> linear activation output
        # we'll use 2 fully connected hidden layers of size 512 instead since we have no conv layers

        self.input_layer = nn.Linear(state_size, hidden_layer_size)
        self.hidden_layer = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.output_layer = nn.Linear(hidden_layer_size, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        # pass through input
        state = self.input_layer(state)
        state = F.relu(state)

        # pass through hidden
        state = self.hidden_layer(state)
        state = F.relu(state)

        # pass through output
        state = self.output_layer(state)

        # last layer has linear activation so return directly
        return state


class DuelingQNetwork(nn.Module):
    """Dueling (Policy) Model."""

    def __init__(self, state_size=37, action_size=4, hidden_layer_size=512, seed=42):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super().__init__()
        self.seed = torch.manual_seed(seed)
        # To mirror the model of the non-dueling network, we'll give the value and advantage layers their own hidden
        # layer
        self.input_layer = nn.Linear(state_size, hidden_layer_size)
        self.value_hidden_layer = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.value_layer = nn.Linear(hidden_layer_size, 1)

        self.advantage_hidden_layer = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.advantage_layer = nn.Linear(hidden_layer_size, action_size)

        # we don't need an output layer explicitly, since dueling DQN has a custom calculation for that

    def forward(self, state):
        """Build a network that maps state -> action values."""
        # pass through input
        state = self.input_layer(state)
        state = F.relu(state)

        # pass through value layers
        value = self.value_hidden_layer(state)
        value = F.relu(value)
        value = self.value_layer(value)

        # pass through advantage layers
        advantage = self.advantage_hidden_layer(state)
        advantage = F.relu(advantage)
        advantage = self.advantage_layer(advantage)

        # Q(S,A) = V(S) + A(S,A) - mean_across_actions(A(S,A))
        Q = value + advantage - advantage.mean(dim=1, keepdim=True)
        return Q
