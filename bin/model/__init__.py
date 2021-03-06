from ..utils import sha1, returns
import torch

class Model(torch.nn.Module):
    def __init__(self, gpu=False):
        """
        Default torch nn model class.

        Args:
        -----
        gpu : bool
            Wheter to use gpu (CUDA)
        """
        super(Model, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() and gpu else 'cpu')
    def set_optimizer(self, optimizer=torch.optim.Adam, optimizer_parameters={'lr' : 1e-3}):
        """
        Set optimizer.

        Args:
        -----
        optimizer : torch.optim class
        optimizer_parameters : dict
            Dict of parameters used in the optimzier initialization.
        """
        self.optimizer = optimizer(
            self.parameters(),
            **optimizer_parameters
        )
    def save(self, path):
        """
        Save model.

        Args:
        -----
        path : str
        """
        if not path[-4:].lower() == '.tar':
            path += '.tar'
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    def load(self, path):
        """
        Load model.

        Args:
        -----
        path : str
        """
        if not path[-4:].lower() == '.tar':
            path += '.tar'
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.eval()

class ActorCritic(Model):
    def __init__(self, observation_space, action_space, optimizer=torch.optim.Adam, optimizer_parameters={'lr' : 1e-3}, gpu=False):
        """
        Actor Critic model.

        Args:
        -----
        observation_space : int
            Size of states.
        action_space : int
            Number of possible actions.
        optimizer : torch.optim class
        optimizer_parameters : dict
            Dict of parameters used in the optimzier initialization.
        gpu : bool
            Wheter to use gpu (CUDA)
        """
        super(ActorCritic, self).__init__(gpu=gpu)
        hidden1 = 10
        self.process = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=observation_space,
                out_features=hidden1,
            ),
            torch.nn.ReLU(inplace=True),
        )
        self.process[0].bias.data.fill_(0)
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=self.process[-2].out_features,
                out_features=action_space
            ),
            torch.nn.Softmax(dim=1),
        )
        self.actor[0].bias.data.fill_(0)
        self.critic = torch.nn.Linear(
            in_features=self.process[-2].out_features,
            out_features=1,
        )
        self.critic.bias.data.fill_(0)
        self.optimizer_name = optimizer.__class__.__name__
        self.optimizer_parameters = optimizer_parameters
        self.set_optimizer(
            optimizer=optimizer,
            optimizer_parameters=optimizer_parameters,
        )
        self.to(self.device)
    def __hash__(self):
        return sha1([
            self.__class__.__name__,
            self.process[0].in_features,
            self.actor[0].out_features,
            self.device,
            self.optimizer_name,
            self.optimizer_parameters,
        ], as_int=True)
    def forward(self, x):
        x = self.process(x.to(self.device))
        return self.actor(x), self.critic(x).view(-1)
    def policy(self, x):
        return self.forward(x)[0]
    def state_value(self, x):
        return self.forward(x)[1]
    def policy_loss(self, action_probabilities, delta, entropy):
        # Shannon entropy
        #   - beta * sum(log(p) * p)
        entropy_loss = - entropy * torch.log(action_probabilities).mul(action_probabilities)
        # Policy loss
        #   - log(p) * delta
        policy_loss = - torch.log(action_probabilities).mul(delta)
        return (policy_loss + entropy_loss).sum()
    def loss(self, states, next_states, actions, rewards, t, entropy, discount_rate, temporal_difference_scale):
        rewards = rewards.to(self.device)
        actual_returns = returns(
            rewards=rewards,
            discount_rate=discount_rate,
        ).to(self.device)
        action_probabilities, _ = self.forward(states.to(self.device))
        action_probabilities = action_probabilities.gather(1, actions.to(self.device)).view(-1)
        return self.policy_loss(
            action_probabilities=action_probabilities,
            delta=actual_returns,
            entropy=entropy,
        ), rewards

class AdvantageActorCritic(ActorCritic):
    def __init__(self, observation_space, action_space, optimizer=torch.optim.Adam, optimizer_parameters={'lr' : 1e-3}, gpu=False):
        """
        Advantage Actor Critic model.

        Args:
        -----
        observation_space : int
            Size of states.
        action_space : int
            Number of possible actions.
        optimizer : torch.optim class
        optimizer_parameters : dict
            Dict of parameters used in the optimzier initialization.
        gpu : bool
            Wheter to use gpu (CUDA)
        """
        super(AdvantageActorCritic, self).__init__(
            observation_space=observation_space,
            action_space=action_space,
            optimizer=optimizer,
            optimizer_parameters=optimizer_parameters,
            gpu=gpu,
        )
    def state_value_loss(self, delta):
        # Sum of squares
        return delta.pow(2).sum()
    def loss(self, states, next_states, actions, rewards, t, entropy, discount_rate, temporal_difference_scale):
        rewards = rewards.to(self.device)
        actual_returns = returns(
            rewards=rewards,
            discount_rate=discount_rate,
        ).to(self.device)
        action_probabilities, expected_returns = self.forward(states.to(self.device))
        action_probabilities = action_probabilities.gather(1, actions.to(self.device)).view(-1)
        advantage = actual_returns - expected_returns
        return self.policy_loss(
            action_probabilities=action_probabilities,
            delta=advantage,
            entropy=entropy,
        ) + self.state_value_loss(
            delta=advantage,
        ), rewards

class TemporalDifferenceAdvantageActorCritic(AdvantageActorCritic):
    def __init__(self, observation_space, action_space, optimizer=torch.optim.Adam, optimizer_parameters={'lr' : 1e-3}, gpu=False):
        """
        Temporal Difference Advantage Actor Critic model.

        Args:
        -----
        observation_space : int
            Size of states.
        action_space : int
            Number of possible actions.
        optimizer : torch.optim class
        optimizer_parameters : dict
            Dict of parameters used in the optimzier initialization.
        gpu : bool
            Wheter to use gpu (CUDA)
        """
        super(TemporalDifferenceAdvantageActorCritic, self).__init__(
            observation_space=observation_space,
            action_space=action_space,
            optimizer=optimizer,
            optimizer_parameters=optimizer_parameters,
            gpu=gpu,
        )
    def loss(self, states, next_states, actions, rewards, t, entropy, discount_rate, temporal_difference_scale):
        rewards = rewards.to(self.device)
        actual_returns = returns(rewards=rewards, discount_rate=discount_rate).to(self.device)
        action_probabilities, expected_returns = self.forward(states.to(self.device))
        action_probabilities = action_probabilities.gather(1, actions.to(self.device)).view(-1)
        next_expected_returns = self.state_value(next_states.to(self.device))
        temporal_difference_advantage = actual_returns - expected_returns + temporal_difference_scale * discount_rate ** t * next_expected_returns
        return self.policy_loss(
            action_probabilities=action_probabilities,
            delta=temporal_difference_advantage,
            entropy=entropy,
        ) + self.state_value_loss(
            delta=temporal_difference_advantage,
        ), rewards
