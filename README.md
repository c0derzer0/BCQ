# Batch-Contrained Deep Q-Learning (BCQ)

Batch-Constrained deep Q-learning (BCQ) is the first batch deep reinforcement learning, an algorithm which aims to learn offline without interactions with the environment.

BCQ was first introduced in our [ICML 2019 paper](https://arxiv.org/abs/1812.02900) which focused on continuous action domains. A discrete-action version of BCQ was introduced in a followup [Deep RL workshop NeurIPS 2019 paper](https://arxiv.org/abs/1910.01708). Code for each of these algorithms can be found under their corresponding folder. 

### Bibtex

```
@inproceedings{fujimoto2019off,
  title={Off-Policy Deep Reinforcement Learning without Exploration},
  author={Fujimoto, Scott and Meger, David and Precup, Doina},
  booktitle={International Conference on Machine Learning},
  pages={2052--2062},
  year={2019}
}
```

```
@article{fujimoto2019benchmarking,
  title={Benchmarking Batch Deep Reinforcement Learning Algorithms},
  author={Fujimoto, Scott and Conti, Edoardo and Ghavamzadeh, Mohammad and Pineau, Joelle},
  journal={arXiv preprint arXiv:1910.01708},
  year={2019}
}
```

### Using BCQ for RSPMN tests (e.g. crossing traffic dataset)
Note: This example test is not rigorous and may have some bugs. The original code is bug free. 
Please look at the code (train_DBCQ, generate_buffer_from_dataset, dargs) in case of issues
```python
from main import train_DBCQ, generate_buffer_from_dataset, dargs
import torch
vars_in_single_time_step = 23
sequence_length = dataset.shape[1]//vars_in_single_time_step
state_vars = 18
action_vars = 4
max_action = 3
num_actions = 16
state_dim = state_vars
action_dim = action_vars
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generate_buffer_from_dataset(dataset, vars_in_single_time_step,
                                 sequence_length, state_vars, action_vars,
                                 device, env='CrossingTraffic', seed=0, buffer_name='testBuffer'
                                 )
                                 
                                 
ddargs = dargs(num_actions, state_dim, 1, 
               env='CrossingTraffic', seed=0, buffer_name='testBuffer', 
               optimizer="Adam", max_timesteps=500000, optimizer_parameters=dictt)
# if an environment exists for interaction
envv = gym.make('Taxi-v3')
ddargs = dargs(num_actions, state_dim, 1, env='Taxi-v3', seed=0, buffer_name='testBuffer', 
               optimizer="Adam", max_timesteps=500000, optimizer_parameters=dictt, 
               do_eval_policy = True, env_made=envv)

model = train_DBCQ(ddargs, device)

# obtaining state meus
import numpy as np
state = [0,0,0,
0,0,0,
1,0,0,
0,1,0,
0,0,0,
0,1,0]
q, imt, i, fq = model.get_q_values(np.array(state))

# obtaining actions
action = model.select_action(np.array([state]), eval=True)
     
```
