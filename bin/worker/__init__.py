from ..utils import Process, Channel, gym_space_size, sha1, rollout
import gym

class Worker(Process):
    def __init__(self, environment, model, model_kwargs={}, shared_model=None, episode_limit=1, rollout_limit=1, rollout_batchsize=-1, discount_rate=1, entropy=1, temporal_difference_scale=0, seed=0, root=None, train=True):
        """
        Worker class for training model on environment.

        Args:
        -----
        environment : str
            Environment name available in Open AI gym.
        model : nn.Module instance
            Model used for training
        model_kwargs : dict
            Kwargs for model initialization
        shared_model : nn.Module instance
            Model identical to other model, which shares memory.
        episode_limit : int
            Number of iterated episodes.
        rollout_limit : int
            Maximum rollout iterations.
        rollout_batchsize : int
            Size of batches used in gradient update.
        discount_rate : float, int
            Discount rate used for discounting future rewards.
        entropy : float, int
            Entropy scaling variable.
        temporal_difference_scale : float, int
            Variable used for scaling temporal difference term in delta varaible in loss function
        seed : int
        root : str or None
            Path used for logging.
        train : bool
            Enable the worker to perform training or validation.
        """
        self.environment = gym.make(environment).env
        self.environment.seed(seed)
        self.environment.action_space.seed(seed)

        # Model
        self.shared_model = shared_model
        model_kwargs['observation_space'] = gym_space_size(self.environment.observation_space)
        model_kwargs['action_space'] = gym_space_size(self.environment.action_space)
        if not hasattr(model, 'device'):
            self.model = model(**model_kwargs)
        else:
            self.model = model

        # Params
        self.channel = Channel()
        self.seed = seed
        self.train = train
        self.episode_limit = episode_limit
        self.rollout_limit = rollout_limit
        self.discount_rate = discount_rate
        self.entropy = entropy
        self.temporal_difference_scale = temporal_difference_scale
        self.rollout_batchsize = rollout_batchsize if rollout_batchsize > 0 else rollout_limit
        super(Worker, self).__init__(root=root, name=self.name)
    def name(self):
        """
        Out:
        ----
        name : str
        """
        return (self.__class__.__name__ if self.train else 'Validator') + str(self.__hash__())
    def __hash__(self):
        return sha1([
            self.environment.__class__.__name__,
            self.model.__hash__(),
            self.seed,
            self.episode_limit,
            self.rollout_limit,
            self.discount_rate,
            self.entropy,
        ], as_int=True)
    def __call__(self):
        self.write(item='Elapsed,Reward')
        self.time.reset()
        if self.train:
            try:
                self.__train__()
            except Exception as e:
                print('Worker error')
                print(str(e), flush=True)
        else:
            try:
                self.__validate__()
            except Exception as e:
                print('Validator error')
                print(str(e), flush=True)
    def __validate__(self):
        if self.shared_model is None:
            model = self.model
        else:
            model = self.shared_model
        self.path.freeze()
        while True:
            # Compute rewards
            rewards_sum = rollout(
                environment=self.environment,
                policy=model.policy,
                limit=self.rollout_limit,
            )[3].sum().item()
            
            # Emit results
            self.channel().send(rewards_sum)

            # Write logs
            self.write(
                item=[self.time.elapsed().total_seconds(), rewards_sum]
            )
        self.path.unfreeze()
    def __train__(self):
        self.path.freeze()
        for episode in range(1, self.episode_limit + 1):
            
            # Rollout
            states, next_states, actions, rewards, _ = rollout(environment=self.environment, policy=self.model.policy, limit=self.rollout_limit)
            rewards_sum = 0
            for batch in zip(
                states.split(self.rollout_batchsize),
                next_states.split(self.rollout_batchsize),
                actions.split(self.rollout_batchsize),
                rewards.split(self.rollout_batchsize),
            ):
                
                # Update model parameters according to shared
                self.model.load_state_dict(self.shared_model.state_dict())
            
                # Clear gradients
                # Important that this happens on local model because 
                # otherwise gradients in shared model could be set
                # to zero while gradients are being used by other
                # process
                self.model.optimizer.zero_grad()
                
                # Calculate loss
                loss, rewards = self.model.loss(
                    *batch,
                    t=states.shape[0],
                    entropy=self.entropy,# ** episode,# (self.episode_limit + 2 - episode) / (self.episode_limit + 1),
                    discount_rate=self.discount_rate,
                    temporal_difference_scale=self.temporal_difference_scale,
                )
                rewards_sum += rewards.sum().item()
                
                # Calculate derivative of loss (backpropagation)
                loss.backward()
                
                # Share gradients with global
                for global_parameters,local_parameters in zip(self.shared_model.parameters(), self.model.parameters()):
                    global_parameters._grad = local_parameters.grad
                
                # Take global optimizer step based on gradients (update shared weights)
                self.shared_model.optimizer.step()
                
            # Emit results
            self.channel().send([episode / (self.episode_limit + 1), rewards_sum])
                
            # Write logs
            self.write(
                item=[self.time.elapsed().total_seconds(), rewards_sum],
            )
        self.path.unfreeze()
