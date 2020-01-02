import torch, os, argparse
from bin.controller import Controller
from bin.worker import Worker
from bin.model import TemporalDifferenceAdvantageActorCritic
from bin.utils import seed

parser = argparse.ArgumentParser(description='A3C PyTorch implementation')
# Worker
parser.add_argument(
    '--w', type=int, default=1, help='amount of workers used in parallel (default: 1)'
)
parser.add_argument(
    '--v', type=int, default=1, help='wheter to use a validation worker for validating alongside training (default: 0)'
)
parser.add_argument(
    '--f', type=int, default=1, help='status frequency in seconds (default: 1 (print every second))'
)
parser.add_argument(
    '--gpu', type=int, default=0, help='wheter to try and use CUDA (default: 0 (do not use CUDA))'
)
parser.add_argument(
    '--out', type=str, default=None, help='path for writing logs (default: None)'
)
parser.add_argument(
    '--seed', type=int, default=0, help='seed (default: 0)'
)
parser.add_argument(
    '--env', type=str, default='CartPole-v1', help='applied Open AI gym environment (default: CartPole-v1)'
)
parser.add_argument(
    '--el', type=int, default=1, help='number of iterated episodes for each worker (default: 1)'
)
parser.add_argument(
    '--rl', type=int, default=1, help='rollout amount for each episode (default: 1)'
)
parser.add_argument(
    '--rb', type=int, default=-1, help='rollout batch size for each rollout (default: -1 (no batches))'
)
# Model
parser.add_argument(
    '--ent', type=float, default=1e-3, help='entropy scaling (default: 1e-3)'
)
parser.add_argument(
    '--dr', type=float, default=0.99, help='discount rate (default: 0.99)'
)
parser.add_argument(
    '--td', type=float, default=0.0, help='temporal difference scaling (default: 0 (no temporal difference))'
)
parser.add_argument(
    '--lr', type=float, default=1e-3, help='optimizser learning rate (default: 1e-3)'
)
# Output
parser.add_argument(
    '--save-model', type=int, default=0, help='wheter to save the model (default: 0)'
)
parser.add_argument(
    '--save-plot', type=int, default=0, help='wheter to save a plot of the results (default: 0)'
)

if __name__ == '__main__':
    args = parser.parse_args()
    seed(seed=args.seed)
    controller = Controller(
        worker=Worker,
        worker_amount=args.w,
        worker_kwargs={
            'environment':args.env,
            'entropy':args.ent,
            'episode_limit':args.el,
            'rollout_limit':args.rl,
            'rollout_batchsize':args.rb,
            'discount_rate':args.dr,
            'temporal_difference_scale':args.td,
            'model':TemporalDifferenceAdvantageActorCritic,
            'model_kwargs':{
                'gpu':args.gpu,
                'optimizer':torch.optim.Adam,
                'optimizer_parameters':{'lr' : args.lr},
            },
        },
        validate=args.v,
        seed=args.seed,
        root=args.out,
    )
    controller(
        status_frequency=args.f,
        save_model=args.save_model
    )
    controller.plot.average(
        window=20,
        save=args.save_plot,
        show=False,
        font_size=25,
        line_width=3,
    )