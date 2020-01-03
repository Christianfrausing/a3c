from ..utils import Process, Channel, Status, sha1
import os
import matplotlib.pyplot as plt
from matplotlib import rcParams

class PlotController:
    def __init__(self, controller):
        """
        Controller plotting class.

        Args:
        -----
        controller : Controller
            Controller instance.
        """
        self.controller = controller
    def average(self, window=10, figure_size=(16,4), font_size=10, line_width=1, save=True, show=True):
        """
        Plot average rewards from workers and a rolling average of the validator rewards.

        Args:
        -----
        window : int
            Window used in rolling average.
        figure_size : tuple
            Size of plot figure.
        font_size : int
            Size of font used in figure.
        line_width : int
            Width of lines in figure.
        save : bool
            Wheter to save an image of the plot.
        show : bool
            Wheter to show the plot.
        """
        rcParams.update({'font.size': font_size})
        fig = plt.figure(figsize=figure_size)
        axes = fig.subplots(nrows=1, ncols=1)
        dfw = sum([worker.read(ext='csv') for worker in self.controller.workers]) / self.controller.worker_amount
        axes.plot(
            dfw.values[:,0],
            dfw.values[:,1],
            label='Average worker rewards',
            alpha=0.5 if self.controller.validate else 1,
            linewidth=line_width,
        )
        if self.controller.validate:
            dfv = self.controller.validator.read(ext='csv').rolling(window=window).mean()
            axes.plot(
                dfv.values[:,0],
                dfv.values[:,1],
                label='Running average validator',
                linewidth=line_width,
            )
        axes.set_xlabel('Time')
        axes.set_ylabel('Rewards')
        axes.spines['right'].set_visible(False)
        axes.spines['top'].set_visible(False)
        axes.legend(loc=4)
        plt.tight_layout()
        if save:
            plt.savefig(self.controller.path() + '/average.png')
        if show:
            plt.show()
        else:
            plt.close(fig)

class Controller(Process):
    def __init__(self, worker, worker_amount=1, worker_kwargs={}, seed=0, root=None, validate=False):
        """
        Controller class used for orchistrating the distributed training of the shared model.

        Args:
        -----
        worker : Worker class
        worker_amount : int
            Amount of workers used in training.
        worker_kwargs : dict
            Kwargs passed in worker initialization.
        seed : int
        root : str or None
            Path used for logging.
        validate : bool
            Wheter to use a validation worker
        """
        # Checks
        assert worker_amount > 0

        # Init
        self.worker = worker
        self.worker_amount = worker_amount
        self.worker_kwargs = worker_kwargs
        self.validate = validate
        self.seed = seed
        self.channel = Channel()
        super(Controller, self).__init__(root=root, name=self.__hash__)

        # Workers
        self.validator = worker(
            seed=seed,
            root=self.path,
            train=False,
            **worker_kwargs,
        )
        # worker_kwargs['model'] = self.validator.model
        self.validator.model.share_memory()
        self.workers = [
            worker(
                shared_model=self.validator.model,
                seed=seed,
                root=self.path,
                **worker_kwargs,
            ) for seed in range(seed + 1, seed + worker_amount + 1)
        ]

        # Params
        self.status = Status(
            header=['Elapsed','Progress [%]', 'Training'] + (['Validation'] if validate else []),
            header_space=5,
            divisor=' | ',
        )
        self.plot = PlotController(controller=self)
    def name(self):
        """
        Out:
        ----
        name : str
        """
        return self.__class__.__name__ + str(self.__hash__())
    def specs(self):
        """
        Out:
        ----
        specs : dict
        """
        return {
            'seed':self.workers[0].seed,
            'episode_limit':self.workers[0].episode_limit,
            'rollout_limit':self.workers[0].rollout_limit,
            'discount_rate':self.workers[0].discount_rate,
            'entropy':self.workers[0].entropy,
            'worker':self.worker.__name__,
            'environment':self.workers[0].environment.__class__.__name__,
            'observation_space':self.workers[0].model.process[0].in_features,
            'action_space':self.workers[0].model.actor[0].out_features,
            'device':str(self.workers[0].model.device),
            'model':self.workers[0].model.__class__.__name__,
            'optimizer':self.workers[0].model.optimizer.__class__.__name__,
            'optimizer_parameters':self.workers[0].model.optimizer_parameters,
        }
    def __hash__(self):
        return sha1({
            'seed': self.seed,
            'worker': self.worker.__class__.__name__,
            'worker_amount':self.worker_amount,
            'worker_kwargs':self.worker_kwargs,
        }, as_int=True)
    def __workers_alive__(self):
        return any([worker.is_alive() for worker in self.workers])
    def __init_logs__(self):
        if not os.path.exists(self.path()):
            os.makedirs(self.path())
        
        # Clear existing files
        self.clear()
        self.validator.clear()
        [worker.clear() for worker in self.workers]
        
        # Create specifications file
        self.write(item=self.specs(), path=self.path() + '/specs')
    def __init_processes__(self):
        # Create channels
        if self.validate:
            self.validator_channel = self.validator.channel.reset()
        self.worker_channels = [worker.channel.reset() for worker in self.workers]
        
        # Apply processes
        if self.validate:
            self.validator.start()
        [worker.start() for worker in self.workers]
    def __call__(self, status_frequency=None, save_model=True):
        """
        Start clntroller.

        Args:
        -----
        status_frequency : int or None
            If None, no status prints are made, otherwise prints are made every
            second described by the variable.
        save_model : bool
            Wheter to save the model when training has finished.
        """
        self.__init_logs__()
        self.__init_processes__()

        # Monitor processes
        if not status_frequency is None:
            self.__monitor__(status_frequency=status_frequency)
        else:
            while self.__workers_alive__():
                pass
            if self.validate:
                self.validator.terminate()
        self.write(item={'runtime':self.time.elapsed().total_seconds()}, path=self.path() + '/specs')

        # Finish processes
        if self.validate:
            self.validator.join()
        [worker.join() for worker in self.workers]
        if save_model:
            self.validator.model.save(path=self.path() + '/model')
        return self.validator.model
    def __monitor__(self, status_frequency):
        count = status_frequency
        worker_progress = [0]*self.worker_amount
        training_status = [0]*self.worker_amount
        validator_status = 0
        self.status.init(device=self.validator.model.device)
        if self.validate:
            while self.__workers_alive__():
                if self.validator_channel.poll(0.1):
                    validator_status = self.validator_channel.recv()
                i = 0
                u = 0
                while i < len(self.worker_channels):
                    try:
                        if self.worker_channels[i].poll(0.1):
                            worker_progress[i], training_status[i] = self.worker_channels[i].recv()
                        i += 1
                        u += 1
                    except EOFError:
                        self.worker_channels.pop(i)
                if count <= self.time.elapsed().total_seconds():
                    self.status(row=[
                        str(self.time.elapsed()).split('.')[0],
                        int(sum(worker_progress) / u * 100),
                        sum(training_status) / u,
                        validator_status,
                    ])
                    count += status_frequency
            self.validator.terminate()
        else:
            while self.__workers_alive__():
                i = 0
                u = 0
                while i < len(self.worker_channels):
                    try:
                        if self.worker_channels[i].poll(0.1):
                            worker_progress[i], training_status[i] = self.worker_channels[i].recv()
                        i += 1
                        u += 1
                    except EOFError:
                        self.worker_channels.pop(i)
                if count <= self.time.elapsed().total_seconds():
                    self.status(row=[
                        str(self.time.elapsed()).split('.')[0],
                        int(sum(worker_progress) / u * 100),
                        sum(training_status) / u,
                    ])
                    count += status_frequency
