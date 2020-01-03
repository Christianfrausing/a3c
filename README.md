# A3C
Implementation of the Asynchronous Advantage Actor Critic (A3C) algorithm in PyTorch.

main.py can be called from terminal where the following args are given:

--w             : amount of workers used in parallel (default: 1)
--v             : wheter to use a validation worker for validating alongside training (default: 0)
--f             : status frequency in seconds (default: 1 (print every second))
--gpu           : wheter to try and use CUDA (default: 0 (do not use CUDA))
--out           : path for writing logs (default: None)
--seed          : seed (default: 0)
--env           : applied Open AI gym environment (default: CartPole-v1)
--el            : number of iterated episodes for each worker (default: 1)
--rl            : rollout amount for each episode (default: 1)
--rb            : rollout batch size for each rollout (default: -1 (no batches))
--ent           : entropy scaling (default: 1e-3)
--dr            : discount rate (default: 0.99)
--td            : temporal difference scaling (default: 0 (no temporal difference))
--lr            : optimizser learning rate (default: 1e-3)
--save-model    : wheter to save the model (default: 0)
--save-plot     : wheter to save a plot of the results (default: 0)
