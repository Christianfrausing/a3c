import utils, gym

class Worker(utils.Process):
    def __init__(self, environment, model, model_kwargs={}, episode_limit=1, rollout_limit=1, rollout_batchsize=-1, discount_rate=1, entropy=1, temporal_difference_scale=0, seed=0, root=None, train=True):
        self.environment = gym.make(environment).env
        self.environment.seed(seed)

        # Model
        model_kwargs['observation_space'] = utils.gym_space_size(self.environment.observation_space)
        model_kwargs['action_space'] = utils.gym_space_size(self.environment.action_space)
        if not hasattr(model, 'device'):
            self.model = model(**model_kwargs)
        else:
            self.model = model

        # Params
        self.channel = utils.Channel()
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
        return (self.__class__.__name__ if self.train else 'Validator') + str(self.__hash__())
    def __hash__(self):
        return utils.sha1([
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
        self.path.freeze()
        while True:
            # Compute rewards
            rewards_sum = utils.rollout(
                environment=self.environment,
                policy=self.model.policy,
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
            states, next_states, actions, rewards, _ = utils.rollout(environment=self.environment, policy=self.model.policy, limit=self.rollout_limit)
            rewards_sum = 0
            for batch in zip(
                states.split(self.rollout_batchsize),
                next_states.split(self.rollout_batchsize),
                actions.split(self.rollout_batchsize),
                rewards.split(self.rollout_batchsize),
            ):
            
                # Clear gradients
                self.model.optimizer.zero_grad()
                
                # Calculate loss
                loss, rewards = self.model.loss(
                    *batch,
                    t=states.shape[0],
                    entropy=self.entropy,
                    discount_rate=self.discount_rate,
                    temporal_difference_scale=self.temporal_difference_scale,
                )
                rewards_sum += rewards.sum().item()
                
                # Calculate derivative of loss (backpropagation)
                loss.backward()
                
                # Take global optimizer step based on gradients
                self.model.optimizer.step()
                
            # Emit results
            self.channel().send([episode / (self.episode_limit + 1), rewards_sum])
                
            # Write logs
            self.write(
                item=[self.time.elapsed().total_seconds(), rewards_sum],
            )
        self.path.unfreeze()
