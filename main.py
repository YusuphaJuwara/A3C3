# @title Main

import os
# import sys
# sys.path.append(os.path.join('G:\My Drive\AI_Robotics\y1-s1-reinforcement-learning\project', 'navigation'))

# import warnings
# warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from typing import Union

import torch.multiprocessing as mp
import threading
import time
from time import sleep

import argparse
from actor_critic import AC_Network
from worker import Worker
from gym_env import GymNav
from shared_adam import SharedAdam

# Logging
import logging
from logger import setLogger

logger = setLogger(name="main", set_level=logging.INFO)

max_episode_length = 500
gamma = 0.9  # discount rate for advantage estimation and reward discounting
lambda_ = 0.95 # GAE hyperparameter
learning_rate = 2.5e-4
spread_messages = False
batch_size = 25

parser = argparse.ArgumentParser(description='Args to be used')
parser.add_argument(
    "--test",
    action='store_true',
    default=False,
    help="train or test (e.g., --test)"
)
parser.add_argument(
    "--load_model",
    action='store_true',
    default=False,
    help="Load model or not (e.g., --load_model)"
)
parser.add_argument(
    "--num_workers",
    type=int,
    default=3,
    help="Set number of available CPU threads/processes"
)
parser.add_argument(
    "--num_agents",
    type=int,
    default=2,
    help="Set number of agents"
)
parser.add_argument(
    "--comm_size",
    type=int,
    default=0,
    help="comm channels"
)
parser.add_argument(
    "--critic",
    type=int,
    default=0,
    help="comm channels"
)
parser.add_argument(
    "--max_epis",
    type=int,
    default=150000,
    help="Max training steps"
)
parser.add_argument(
    "--save_dir",
    type=str,
    default="models",
    help="Folder for saving during training"
)
parser.add_argument(
    "--comm_gaussian_noise",
    type=float,
    default=0,
    help="Random guassian noice added to communication"
)
parser.add_argument(
    "--comm_delivery_failure_chance",
    type=float,
    default=0,
    help="The percentage of delivery failure"
)
parser.add_argument(
    "--comm_jumble_chance",
    type=float,
    default=0,
    help="The percentage of jumble chance"
)
parser.add_argument(
    "--param_search",
    type=str,
    default="40,relu,80,40",
    help="network configs. [size_of_comm_HL, activ_fn, size_of_actor_and_critic_HL1, size_of_actor_and_critic_HL2, ...]"
)
args = parser.parse_args()
number_of_agents = args.num_agents
comm_size = args.comm_size
amount_of_agents_to_send_message_to = number_of_agents - 1

network_configs = args.param_search.split(",")
network_configs[0] = int(network_configs[0])
for i in range(2, len(network_configs)):
    network_configs[i] = int(network_configs[i])

display = False

if args.test:
    # args.load_model = True
    args.num_workers = 1
    display = True
    learning_rate = 0
    args.max_epis += 1000
    batch_size = max_episode_length + 1
    
logger.debug(f"Args: {args}")

env = GymNav(number_of_agents=number_of_agents)
state_size = env.agent_observation_space
s_size_central = env.central_observation_space
action_size = env.agent_action_space
reward_range = env.reward_range
env.close()

logger.debug(f"""
             state_size: {state_size}, 
             s_size_central: {s_size_central},
             action_size: {action_size}, 
             reward_range: {reward_range}
             """)

critic_action = False
critic_comm = False
if args.critic == 1 or args.critic == 3:
    critic_action = True
if args.critic == 2 or args.critic == 3:
    critic_comm = True

# Create a directory to save models and episode playback gifs
# exist_ok = True to not raise an error if the dir(s) exist(s)
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir, exist_ok=True)

comm_size_input = amount_of_agents_to_send_message_to * comm_size

# Generate global network
global_network = \
    AC_Network(
        s_size=state_size, 
        s_size_central=s_size_central, 
        number_of_agents=number_of_agents,
        a_size=action_size, 
        comm_size_input=comm_size_input,
        comm_size_output=comm_size_input if spread_messages else comm_size,
        critic_action=critic_action, 
        critic_comm=critic_comm,
        paramSearch=network_configs
            ) 
    
global_network.share_memory() # Share memory between processes

model_path = os.path.join(args.save_dir, "actor_critic_1.pkl")
if args.load_model:
    global_network.load_state_dict(torch.load(model_path))
    
# Optimizers
###########################################################
# For policy and value networks
# global_weights = list(global_network.obs_hidden.parameters()) + \
#                 list(global_network.central_hidden.parameters()) + \
#                 list(global_network.value.parameters()) + \
#                 list(global_network.policy.parameters())

optimizer = SharedAdam(global_network.parameters(), lr=learning_rate, 
                    betas=(0.92, 0.999))
optimizer.share_memory() # Share memory between processes
    
# Better to update the actor and critic's weights differently for network stability; 
# recommended by Sergey Levine of UC Berkeley.

# policy_lr = 3e-6 # slow to let critic digest ...
# value_lr = 1e-4
# msg_lr = 1e-4

# policy_weights = list(global_network.obs_hidden.parameters()) + \
#                 list(global_network.policy.parameters()) 
# policy_optimizer = SharedAdam(policy_weights, lr=policy_lr,
#                     betas=(0.92, 0.999))
# policy_optimizer.share_memory() # Share memory between processes
# # lr_scheduler = optim.lr_scheduler.StepLR(policy_optimizer, step_size=1, gamma=0.9)

# value_weights = list(global_network.central_hidden.parameters()) + \
#                 list(global_network.value.parameters())
# value_optimizer = SharedAdam(value_weights, lr=value_lr,
#                     betas=(0.92, 0.999))
# value_optimizer.share_memory() # Share memory between processes

# comm_weights = list(global_network.hidden_comm.parameters()) + \
#                 list(global_network.comm_fn.parameters())
# comm_optimizer = SharedAdam(comm_weights, lr=msg_lr, 
#                     betas=(0.92, 0.999))
# comm_optimizer.share_memory() # Share memory between processes
####################################################################

global_episodes = mp.Value('i', 0)
global_best_rew = mp.Value('d', 0.0)

workers = []
# Create worker classes
for i in range(args.num_workers):
    workers.append(Worker(game=GymNav(number_of_agents=number_of_agents), 
                        name=i, 
                        s_size=state_size, 
                        s_size_central=s_size_central,
                        a_size=action_size, 
                        number_of_agents=number_of_agents, 
                        global_AC=global_network,
                        optimizer=optimizer,
                        save_dir=args.save_dir,
                        model_path=model_path,
                        global_episodes=global_episodes, 
                        amount_of_agents_to_send_message_to=amount_of_agents_to_send_message_to,
                        display=display and i == 0, 
                        comm=(comm_size != 0),
                        comm_size_per_agent=comm_size, 
                        spread_messages=spread_messages,
                        critic_action=critic_action, 
                        critic_comm=critic_comm,
                        comm_delivery_failure_chance=args.comm_delivery_failure_chance,
                        comm_gaussian_noise=args.comm_gaussian_noise,
                        comm_jumble_chance=args.comm_jumble_chance,
                        paramSearch=network_configs,
                        max_episode_length=max_episode_length, 
                        max_num_of_episodes=args.max_epis, 
                        gamma=gamma, 
                        lambda_=lambda_, 
                        batch_size=batch_size
                        )
                   )


# This is where the asynchronous magic happens.
# Start the "work" process for each worker in a separate process/thread.
worker_threads = []
for worker in workers:
    worker_work = lambda: worker.work()
    t = threading.Thread(target=(worker_work))
    t.start()
    sleep(0.5)

    worker_threads.append(t)

for t in worker_threads:
    t.join()