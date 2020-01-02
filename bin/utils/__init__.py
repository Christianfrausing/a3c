import gym, torch, hashlib, os, datetime, json, multiprocessing, numpy, random, inspect
from pandas import read_csv

def gym_space_size(x):
    if isinstance(x, gym.spaces.Discrete):
        return x.n
    elif isinstance(x, gym.spaces.Box):
        return x.shape[0]
def sha1(parameters, as_int=False):
    checksum = hashlib.sha1()
    checksum.update(str(parameters).encode('utf-8'))
    if as_int:
        return int(checksum.hexdigest(), 16)
    else:
        return checksum.hexdigest()
def remove(path):
    try:
        os.remove(path)
    except OSError:
        print('Could not remove path %s' %path)
def returns(rewards, discount_rate=1):
    """
    Calculating the expected (discounted) returns.

    Args:
    -----
    rewards : tensor
        Rewards computed by environment rollout.
    discount_rate : float, int
        Discount rate used for discounting future rewards.
    
    Out:
    ----
    returns : tensor
        Computed discounted rewards (returns).
    """
    returns = rewards.clone()
    for t in range(len(rewards) - 2, -1, -1):
        returns[t] = rewards[t] + discount_rate * returns[t+1]
    return returns
def rollout(environment, policy, limit=1):
    """
    Computing rollout by taking steps in environment without
    updating policy.

    Args:
    -----
    environment : Open AI gym environment
        Open AI gym environment instance.
    policy : function
        Function which takes states as input and returns
        action probabilities.
    limit : int
        Limit which determines when the size of the rollout.
        If the environment step results in completion, this
        event also restricts the size of the rollout.
    
    Out:
    ----
    rollout : tuple
        Tuple of states, next_states, actions and rewards 
        according to the steps taken in the environment.
    """
    states,next_states,actions,rewards = [],[],[],[]
    state = environment.reset()
    t,complete = 0,False
    while not complete and t < limit:
        states.append(state)
        with torch.no_grad():
            probabilities = policy(torch.FloatTensor([states[-1]]))
        actions.append(torch.distributions.categorical.Categorical(probs=probabilities).sample())
        state, reward, complete, _ = environment.step(actions[-1].item())
        next_states.append(state)
        rewards.append(reward)
        t += 1
    return (
        torch.FloatTensor(states),
        torch.FloatTensor(next_states),
        torch.LongTensor(actions).view(-1,1),
        torch.FloatTensor(rewards),
        t,
    )
def seed(seed):
    """
    Seed setting function used to control the seed for the
    random initialization.

    Args:
    -----
    seed : int
    """
    # Seed settings copied from
    # https://github.com/pytorch/pytorch/issues/7068
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    numpy.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class Time:
    def __init__(self):
        """
        Time class used for measuring elapsed time in process.
        """
        self.reset()
    def reset(self):
        """
        Resets time origin.
        """
        self.origin = datetime.datetime.now()
    def elapsed(self):
        """
        Calculates elapsed time between origin and now.
        """
        return datetime.datetime.now() - self.origin

class Path:
    def __init__(self, root=None, name=None):
        """
        Path class used for determining logging path in process.

        Args:
        -----
        root : str or None
            Path used for logging.
        name : str or None
            Name used for logging.
        """
        self.name = name
        self.root = root
        self.unfreeze()
    def freeze(self):
        """
        Freezes path (stores path in variable).
        """
        self._path = self.__call__()
    def unfreeze(self):
        """
        Unfreezes path (set _path variable to None).
        """
        self._path = None
    def __call__(self):
        """
        Out:
        ----
        path : str
            Path for logging.
        """
        if self._path is None:
            suffix = '/'
            if isinstance(self.name, str):
                suffix += self.name
            elif callable(self.name):
                suffix += str(self.name())
            if isinstance(self.root, str):
                return self.root + suffix
            elif callable(self.root):
                return self.root() + suffix
            elif self.root is None:
                return os.path.dirname(__file__) + '/logs' + suffix
        else:
            return self._path

class Channel:
    def __init__(self):
        """
        Channel class used for communication between processes.
        """
        self.writer = None
    def __call__(self):
        """
        Out:
        ----
        writer : pipe instance
            Pipe instance which allows sending.
        """
        return self.writer
    def reset(self):
        """
        Reset communication for process.

        Out:
        ----
        reader : pipe instance
            Pipe instance which allows recieving.
        """
        reader,writer = multiprocessing.Pipe(duplex=False)
        self.writer = writer
        return reader

class Status:
    def __init__(self, header, header_space=5, divisor=' | '):
        """
        Path class used for determining logging path in process.

        Args:
        -----
        header : list
            List of items displayed in status header.
        header_space : int
            Amount of spaces ' ' used between headers.
        divisor : str
            String which is used to join header elements.
        """
        self.header = header
        self.header_space = ' '*header_space
        self.divisor = divisor
    def __call__(self, row):
        """
        Prints the specifed row according to status settings.

        Args:
        -----
        row : list
            List of row containing elements which matches the headers.
        """
        for i in range(len(row)):
            header_length = len(self.header[i] + self.header_space)
            row_length = len(str(row[i]))
            if isinstance(row[i], str):
                if row_length > header_length:
                    row[i] = row[i][:header_length]
                else:
                    row[i] += ' '*(header_length - row_length)
            else:
                if row_length > header_length:
                    row[i] = eval("'{:._E}'.replace('_', str(header_length - 6 if (header_length - 6) > 0 else 0)).format(row[i])")
                else:
                    row[i] = str(row[i]) + ' '*(header_length - row_length)
        print(self.divisor.join(row), flush=True)
    def init(self, device):
        """
        Initialize status by intro and header.

        Args:
        -----
        device : device
            Torch device.
        """
        header = self.divisor.join([header + self.header_space for header in self.header])
        print('Controller running on %s' %(str(device).upper()), flush=True)
        print(flush=True)
        print(header, flush=True)
        print('_'*(len(header)), flush=True)

class Process(torch.multiprocessing.get_context('spawn').Process):
    # The spawn context is used in order to allow GPU usage
    def __init__(self, name='', root=None):
        """
        Process class used for multiprocessing.

        Args:
        -----
        name : str
            Process name.
        root : str or None
            Process root location.
        """
        super(Process, self).__init__(target=self.__call__)
        self.root = root
        self.time = Time()
        self.path = Path(root=root, name=name)
    def write(self, item, path=''):
        """
        Write item to file. If the item variable is a str or list
        the data is written to the csv file, otherwise if the
        item is a dict is is written to the json file.

        Args:
        -----
        item : str, list or dict
            Item written to disc.
        path : str
            Path for which item is written, including file extension.
        """
        if not path:
            path = self.path()
        # csv
        if isinstance(item, str):
            with open(path + '.csv', 'a') as csv_file:
                csv_file.write(item)
        elif isinstance(item, list):
            with open(path + '.csv', 'a') as csv_file:
                csv_file.write('\n' + ','.join([str(i) for i in item]))
        # json
        elif isinstance(item, dict):
            path += '.json'
            if os.path.exists(path):
                with open(path, 'r') as f:
                    temp = json.load(f)
                    temp.update(item)
                    item = temp
            with open(path, 'w') as f:
                json.dump(item, f)
    def clear(self):
        """
        Delete both csv and json file if they exists.
        """
        path = self.path()
        if os.path.exists(path + '.csv'):
            remove(path + '.csv')
        if os.path.exists(path + '.json'):
            remove(path + '.json')
    def __call__(self):
        return
    def read(self, ext='csv'):
        """
        Read either csv or json file.

        Args:
        -----
        ext : str
            Either csv or json.
        
        Out:
        ----
        file : pandas dataframe or dict
            - csv : pandas dataframe.
            - json : dict.
        """
        if ext == 'csv':
            return read_csv(self.path() + '.' + ext)
        elif ext == 'json':
            with open(self.path() + '/specs.' + ext) as f:
                return json.load(f)
    def exists(self, ext=''):
        """
        Determine if file exists on process path.

        Args:
        -----
        ext : str
            Either csv or json.
        """
        return os.path.exists(self.path() + ext)
