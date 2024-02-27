# @title Worker class in PyTorch

from collections import deque
import random
from time import time
import tqdm

import matplotlib.pyplot as mpl
from time import sleep
import math

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
# from torch.distributions import normal
from torch.utils.tensorboard import SummaryWriter

from actor_critic import AC_Network
from gym_env import GymNav
from helper import one_hot_encoding
from shared_adam import SharedAdam
from replay_buffer import ReplayBuffer

from typing import Union, List, Tuple

# Logging
import logging
from logger import setLogger

logger = setLogger(name="Worker", set_level=logging.INFO)


# Worker class in PyTorch
class Worker:
    def __init__(self, 
                game: GymNav, 
                name: int, 
                s_size: list[int], 
                s_size_central: list[int], 
                a_size: int, 
                number_of_agents: int, 
                global_AC: AC_Network, 
                optimizer: Union[torch.optim.Optimizer, SharedAdam],
                save_dir: str,
                model_path: str,
                global_episodes: int, 
                amount_of_agents_to_send_message_to: int,
                display: bool = False, 
                comm: bool = False, 
                comm_size_per_agent: int = 0, 
                spread_messages: bool = True,
                critic_action: bool = False, 
                critic_comm: bool = False,
                comm_delivery_failure_chance: float = 0.0, 
                comm_gaussian_noise: float = 0.0, 
                comm_jumble_chance: float = 0.0,
                paramSearch: tuple[int, str, int, int] = [40, "relu", 80, 40],
                max_episode_length: int = 500, 
                max_num_of_episodes: int = 150_000,
                gamma: float = 0.9, 
                lambda_: float = 0.95,
                batch_size: int = 25
            ) -> None:
        """This is the `Worker` class, where each worker has a local copy of the global network(s) 
        -- Actor, Critic, and Communication Networks. These networks are shared across all workers.
        
        Args:
            - game: The `Gymnasium`  game environment
            - name: The name/index of the worker
            - s_size: The size of the state space -> e.g., list[6]
            - s_size_central: The size of the state space used as an input to the ` Centralized Critic` -> e.g., list[8]
            - a_size: The size of the action space -> 5
            - number_of_agents: The number of agents in the environment
            - global_AC: The global network that each worker copies and updates
            - optimizer: The optimizer for the global network
            - save_dir: The directory to save the models to
            - model_path: The model name in the save_dir
            - global_episodes: The max number of episodes to train
            - amount_of_agents_to_send_message_to: The number of agents to send message to -> range[0, number_of_agents-1]
            - display: Whether to display the game or not -- only Worker_0 is displayed in training mode
            - comm: Whether to use communication between agents or not
            - comm_size_per_agent: The number of communication channels or msg size each agent can sent to others
            - spread_messages: Whether to spread messages or not
            - critic_action: Whether to use the action critic or not ... ?
            - critic_comm: Whether to use the communication critic or not ... ?
            - comm_delivery_failure_chance: A probability to fail to deliver a message to an agent
            - comm_gaussian_noise: A probability to add gaussian noise to the messages
            - comm_jumble_chance: A probability to jumble the messages. Akin to communication (WIFI), 
            this means that the msgs sent from other agents could mixed up. So, the agent should agregate (sum, mean) the msgs.
            - paramSearch: The parameters to be used in the hidden layers, and acttivations of the hidden layers.
            E.g., paramSearch = "[size_of_comm_HL, activ_fn, size_of_actor_and_critic_HL1, size_of_actor_and_critic_HL2, ...]".
            - `max_episode_length`: maximum length of an episode
            - `max_num_of_episodes`: maximum number of episodes to train for
            - `gamma`: discount factor used to calculate TD-targets (deltas). Default to 0.9
            - `lambda_`: decay rate for calculating the GAE. Default to 0.95
            - `batch_size`: batch size for training. Default to 25
            
        Returns:
            - None
            
        E.g.:
        ```python
        for i in range(args.num_workers):
            workers.append(Worker(game=GymNav(number_of_agents=number_of_agents), 
                                name=i, 
                                s_size=[6], 
                                s_size_central=[8], 
                                a_size=5, 
                                number_of_agents=4, 
                                global_AC=global_network, 
                                optimizer=optimizer,
                                save_dir="models", 
                                model_path="models/actor_critic_1.pkl", 
                                global_episodes=1_000_000, 
                                amount_of_agents_to_send_message_to=number_of_agents-1, 
                                display=True and i == 0, 
                                comm=(comm_size != 0), 
                                comm_size_per_agent=20, 
                                spread_messages=False,
                                critic_action=0, 
                                critic_comm=0, 
                                comm_delivery_failure_chance=0.0,
                                comm_gaussian_noise=0.2, 
                                comm_jumble_chance=0.0,  
                                paramSearch=[40, "relu", 80, 40],
                                max_episode_length = 500, 
                                max_num_of_episodes = 150_000,
                                gamma = 0.9, 
                                lambda_ = 0.95,
                                batch_size = 25
                                )
                   )
        ```
        """
        
        torch.autograd.set_detect_anomaly(True)
        
        self.name = "worker_" + str(name)
        self.is_chief = self.name == 'worker_0'
        logger.info(f"Worker name: {self.name}")
        
        # Write to tensorboard
        self.writer = SummaryWriter(f'tb_logs/{self.name}')

        self.number = name
        self.number_of_agents = number_of_agents
        self.save_dir = save_dir
        self.model_path = model_path
        self.global_AC = global_AC
        self.global_episodes = global_episodes
        self.amount_of_agents_to_send_message_to = amount_of_agents_to_send_message_to
        self.critic_action = critic_action
        self.critic_comm = critic_comm
        self.paramSearch = paramSearch
        self.comm_size_input = amount_of_agents_to_send_message_to * comm_size_per_agent
        self.comm_size_output = self.comm_size_input if spread_messages else comm_size_per_agent
        self.comm = comm
        self.display = display
        self.message_size = comm_size_per_agent
        self.spread_messages = spread_messages
        self.spread_rewards = False
        self.comm_delivery_failure_chance = comm_delivery_failure_chance
        self.comm_gaussian_noise = comm_gaussian_noise
        self.comm_jumble_chance = comm_jumble_chance
        
        self.max_episode_length = max_episode_length
        self.max_num_of_episodes = max_num_of_episodes
        self.gamma = gamma 
        self.lambda_ = lambda_ 
        
        # Env set-up
        self.env = game
        self.s_size = s_size
        self.s_size_central = s_size_central
        self.a_size = a_size
        
        # Create the local network
        self.local_AC = self.getNet()
        # Copy the global parameters to local network
        # self.local_AC.load_state_dict(self.global_AC.state_dict())
        
        self.optimizer = optimizer
        
        ####################################
        self.episode_rewards = deque(maxlen=2 * number_of_agents)
        self.episode_lengths = deque(maxlen=2 * number_of_agents)
        self.episode_loss = deque(maxlen=2 * number_of_agents)
        ####################################
        
        self.episode_buffer = ReplayBuffer(num_agents=self.number_of_agents,
                                        buffer_capacity=batch_size,
                                        state_size=self.s_size[0],
                                        state_size_central=self.s_size_central[0],
                                        action_size=self.a_size,
                                        msg_size=self.message_size
                                    )
        
        if self.comm_gaussian_noise != 0:
            # Initialize the normal distribution for the comm noise to be injected into the msg.
            self.normal = torch.distributions.normal.Normal(loc=0, scale=self.comm_gaussian_noise)
        
    def getNet(self) -> AC_Network:
        """_summary_:
            Initializes and returns the network of the worker thread/process
            
        Args:
            - None

        Returns:
            - AC_Network: the model architecture of the worker thread/process
            
        E.g.:
        ```python
        self.local_AC = self.getNet()
        ```
        """
        return AC_Network(s_size=self.s_size, 
                        s_size_central=self.s_size_central, 
                        number_of_agents=self.number_of_agents,
                        a_size=self.a_size, 
                        comm_size_input=self.comm_size_input,
                        comm_size_output=self.comm_size_output,
                        critic_action=self.critic_action, 
                        critic_comm=self.critic_comm,
                        paramSearch=self.paramSearch
                        )

    def work(self) -> None:
        """
        _summary_:
        This is where the asynchronous magic happens. Start the "work" process for each worker in a separate process/thread.
        Train the 3 networks -- Actor, Critic, and Communication -- for each worker and update the global network.
        
        Args:
            None
            
        Returns:
            None
            
        E.g.:
        ```python
        worker_threads = []
        
        # Remember the worker in the __init__ method?
        for worker in workers:
            worker_work = lambda: worker.work()
            t = threading.Thread(target=(worker_work))
            t.start()
            sleep(0.5)

            worker_threads.append(t)

        for t in worker_threads:
            t.join()
        ```
        """
        
        logger.info("Starting worker " + str(self.number))

        while self.global_episodes.value < self.max_num_of_episodes:
            
            # Copy the global parameters to local network at the start of every episode
            self.local_AC.load_state_dict(self.global_AC.state_dict())

            # This holds, for each agent, the agents that sent it a message. 
            # episode_comm_maps = [[] for _ in range(self.number_of_agents)]
            episode_comm_maps = torch.empty((self.number_of_agents, self.number_of_agents-1), dtype=torch.int32)
                        
            for i in range(self.number_of_agents):
                comm_map = list(range(self.number_of_agents))
                
                # Indices of all the agents except the current agent
                comm_map.remove(i)
                
                # Each agent has a list of the indices of the agents they send messages to
                # Shape: (number_of_agents, amount_of_agents_to_send_message_to=number_of_agents-1)
                episode_comm_maps[i] = torch.tensor(comm_map)

            curr_comm = torch.zeros((self.number_of_agents, 
                                     self.amount_of_agents_to_send_message_to * self.message_size),
                                     dtype=torch.float, requires_grad=True)
                    
                    
            # The current_screen = List[List[int]] = arrayed_current_screen_central
            # I.e., it returns the obs for all the agents -> batch of agents' observations
            current_screen, info = self.env.reset()
            arrayed_current_screen_central = info["state_central"]
            
            current_screen = torch.tensor(current_screen, dtype=torch.float)
            arrayed_current_screen_central = torch.tensor(arrayed_current_screen_central, dtype=torch.float)
            
            if self.is_chief and self.display:
                self.env.render()
                
            # When terminal is true, the episode len might be less than the buffer capacity, where
            # the rest is zeros, false, ...
            terminal_len = 0 
            
            # Run until episode length is reached
            for episode_step_count in range(self.max_episode_length):
                
                terminal_len += 1
                    
                # Sample an action, log probability, and entropy for each agent from $\pi_{\theta_{OLD}}$ for each worker;
                # -> not used for gradient calculations
                # with torch.no_grad():
                actions, action_log_prob, action_entropy = self.local_AC.action_prob_entropy(
                                                                                obs=current_screen, 
                                                                                curr_comm=curr_comm, 
                                                                                eval_model=False
                                                                                   )

                # Run the local (message) network to get the msg to send to the other agents
                message = self.local_AC.getMessage(obs=current_screen)
                
                # message gauss noise
                if self.comm_gaussian_noise != 0:
                    message = message + self.normal.sample(message.shape)

                # Watch environment
                next_screen, reward, terminal, info = self.env.step(actions)
                arrayed_next_screen_central = info["state_central"]
                
                next_screen = torch.tensor(next_screen, dtype=torch.float)
                arrayed_next_screen_central = torch.tensor(arrayed_next_screen_central, dtype=torch.float)

                if self.critic_action:
                    for agent in range(self.number_of_agents):
                        
                        acts = actions.tolist()
                        logger.debug(f"acts: {acts} \n actions[0:agent]: {acts[0:agent]} \n actions[agent + 1:]: {acts[agent + 1:]}")
                
                        
                        actions_one_hot = \
                            one_hot_encoding(acts[0:agent] + acts[agent + 1:],
                                            self.a_size)
                        logger.debug(f"actions_one_hot: {actions_one_hot}")
                            
                        arrayed_current_screen_central[agent] = \
                            arrayed_current_screen_central[agent] + torch.tensor(actions_one_hot)
                            
                if self.critic_comm:
                    for agent in range(self.number_of_agents):
                        arrayed_current_screen_central[agent] = \
                            arrayed_current_screen_central[agent] + curr_comm[agent] 
                    
                # TODO: handle this part
                # this_turns_comm_map = []
                # for i in range(self.number_of_agents):
                #     # 50% chance of no comms
                #     surviving_comms = list(range(self.number_of_agents))
                #     surviving_comms.remove(i)
                #     for index in range(len(surviving_comms)):
                #         if random.random() < self.comm_delivery_failure_chance:  # chance of failure comms
                #             surviving_comms[index] = -1
                #     episode_comm_maps[i].append(surviving_comms)
                #     this_turns_comm_map.append(surviving_comms)
                curr_comm = self.output_mess_to_input_mess(message=message, 
                                                           episode_comm_maps=episode_comm_maps
                                                           )

                # TODO: jumbles comms
                # if self.comm_jumble_chance != 0:
                #     for i in range(self.number_of_agents):
                #         joint_comm = [0] * self.message_size
                #         for index in range(len(curr_comm[i])):
                #             joint_comm[index % self.message_size] += curr_comm[i][index]
                #         jumble = False
                #         for index in range(len(curr_comm[i])):
                #             if index % self.message_size == 0:
                #                 # only jumble messages that got received
                #                 jumble = curr_comm[i][index] != 0 and random.random() < self.comm_jumble_chance
                #             if jumble:
                #                 curr_comm[i][index] = joint_comm[index % self.message_size]

                if self.is_chief and self.display:
                    self.env.render()
                    
                self.episode_rewards.append(sum(reward) if self.spread_rewards else reward)
                                                             
                # Update the networks if buffer == batch_size or terminal using the experience rollout
                transition = {'state': current_screen,
                              'state_central': arrayed_current_screen_central,
                              'msg_sent': message,
                              'msg_rec': curr_comm,
                              'action': actions,
                              'action_log_prob': action_log_prob,
                              'reward': reward,
                              'terminal': terminal,
                              }
                
                current_screen = next_screen
                arrayed_current_screen_central = arrayed_next_screen_central
                
                # Store the transition in the replay buffer and check if the buffer is full
                is_full = self.episode_buffer.store(**transition)
                
                # If the episode is over or if the buffer is full then update the network(s)
                if terminal or is_full or episode_step_count == self.max_episode_length - 1:
                                        
                    # Note that values are batch_size + 1 bec the observations are batch_size + 1
                    # I.e., rewards + gamma * values[1:] * terminals - values[:-1] -> Shape: (batch_size ...)
                    # So, add the last observation to the buffer
                    self.episode_buffer.store_last_observation(arrayed_next_screen_central)
                    
                    for agent in range(self.number_of_agents):
                    
                        # Calculate the losses
                        loss = self.calc_losses(agent, terminal_len)
                    
                        # Update the global network with losses   
                        self.update_network(loss, message, curr_comm)
                        
                        # Save statistics for TensorBoard
                        self.episode_loss.append(loss.detach().numpy())

                    # Copying global networks to local networks
                    self.local_AC.load_state_dict(self.global_AC.state_dict())
                    
                    # Clear the buffer for a new rollout
                    self.episode_buffer.clear_buffer() 
                    
                    terminal_len = 0

                # If both prey and predator have acknowledged game is over, then break from episode
                if terminal:
                    break
                
            self.episode_lengths.append(episode_step_count)
            
            # Periodically save statistics.
            if self.global_episodes.value % self.episode_lengths.maxlen == 0 \
                or self.global_episodes.value == self.max_num_of_episodes - 1:

                # Save statistics for TensorBoard
                mean_length = np.mean(self.episode_lengths)
                mean_reward = np.mean(self.episode_rewards)
                mean_loss = np.mean(self.episode_loss)
                
                logger.info(f"Global episodes @ {self.global_episodes.value + 1}/{self.max_num_of_episodes}")
                logger.info(f"""\n\t mean_reward: {mean_reward},
                            \n\t mean_value_loss: {mean_loss}
                            """)
                
                self.writer.add_scalar(tag='Mean Episode Length', scalar_value=mean_length, global_step=self.global_episodes.value)
                self.writer.add_scalar(tag='Mean Episode Loss', scalar_value=mean_loss, global_step=self.global_episodes.value)
                self.writer.add_scalar(tag='Mean Episode Reward', scalar_value=mean_reward, global_step=self.global_episodes.value)
                self.writer.flush() # Call flush() method to make sure that all pending events have been written to disk.

            # Update global episode count
            with self.global_episodes.get_lock():
                self.global_episodes.value += 1
                
                # Save current model
                if self.global_episodes.value % 1_000 == 0 \
                    or self.global_episodes.value == self.max_num_of_episodes:
                    self.local_AC.save_checkpoint(model_name=self.model_path)

        # Close the environment and writer.
        self.env.close()
        self.writer.close()
        
    @torch.no_grad()
    def calculate_advantage_and_gae(self, 
                                    rewards: Tensor, 
                                    values: Tensor, 
                                    terminals: Tensor
                                    ) -> Tensor:
        """
        - Calculates the TD target (deltas) -> delta = r_t + gamma * V(s_t+1) - V(s_t)
        - and Generalized Advantage Estimation (GAE) -> A(s_t) = delta_t + gamma * lambda * A(s_t+1)
        
        Ref: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/rl/ppo/gae.py
        
        Args:
            - `rewards`: The rewards. Shape: (batch_size)
            - `values`: The Critic values. Shape: (batch_size + 1)
            - `terminals`: The terminals. Shape: (batch_size)
            
        Returns:
            - `advantages`: The GAE values. Shape: (batch_size)
            
        E.g.:
        ```python
            >>> import torch
            >>> rewards = torch.tensor([1, 2, 3, 4])
            >>> values = torch.tensor([1, 2, 3, 4])
            >>> terminals = torch.tensor([False, False, False, True])
            
            >>> advantages = self.calculate_advantage_and_gae(rewards, values, terminals)
        ```
        """
        
        # Calculate TD errors (deltas) -> delta = r_t + gamma * V(s_t+1) - V(s_t)
        # Shape: (batch_size)
        deltas = rewards + self.gamma * values[1:] * (~terminals) - values[:-1] # ~term = 1-term

        # Calculate advantages using GAE -> A(s_t) = delta_t + gamma * lambda * A(s_t+1)
        advantages = self.calculate_gae(deltas, terminals)

        logger.debug(f"""\ncalculate_advantage_and_gae: 
                    \n rewards: {rewards}
                    \n values: {values}
                    \n terminals: {terminals}
                    \n deltas: {deltas} 
                    \n advantages: {advantages}
                    """)

        return advantages
        
    def calculate_gae(self, 
                      deltas: Tensor,  
                      terminals: Tensor
                      ) -> Tensor:
        """Calculates the Generalized Advantage Estimation (GAE) -> A(s_t) = delta_t + gamma * lambda * A(s_t+1).

        Args:
            `deltas` (Tensor[FloatTensor]): TD errors (deltas) -> delta = r_t + gamma * V(s_t+1) - V(s_t). Shape: (batch_size).
            `terminals` (Tensor[BoolTensor]): whether or not the episode was terminated. Shape: (batch_size).

        Returns:
            `advantages` (Tensor[FloatTensor]): The GAE values. Shape: (batch_size).
            
        Example:
        ```python
        >>> import torch
        >>> deltas = torch.tensor([1, 2, 3, 4])
        >>> terminals = torch.tensor([False, False, False, False])
        
        >>> advantages = self.calculate_gae(deltas, terminals)
        ```
        """
        gae = torch.zeros_like(deltas)
        running_add = 0
        
        # Notice the reversed()
        for t in reversed(range(len(deltas))):
            # Calculate advantages using GAE -> A(s_t) = delta_t + gamma * lambda * A(s_t+1)
            running_add = deltas[t] + running_add * self.gamma * self.lambda_ * (~terminals[t])
            gae[t] = running_add

        return gae
    
    @staticmethod
    def normalize_advantage(advantages: Tensor):
        """Normalizes the advantages using the mean and standard deviation of the advantages.

        Args:
            advantages (Tensor[FloatTensor]): the GAE that will be normalized.

        Returns:
            Tensor[FloatTensor]: the normalized GAE
        """
        return (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    @staticmethod
    def clipped_ppo_loss(log_pi: torch.Tensor, 
                         sampled_log_pi: torch.Tensor,
                         advantage: torch.Tensor, 
                         clip: float
                         ) -> torch.Tensor:
        """
        ## PPO Loss

        Here's how the PPO update rule is derived.

        We want to maximize policy reward
        $$\max_\theta J(\pi_\theta) =
        \mathop{\mathbb{E}}_{\tau \sim \pi_\theta}\Biggl[\sum_{t=0}^\infty \gamma^t r_t \Biggr]$$
        where $r$ is the reward, $\pi$ is the policy, $\tau$ is a trajectory sampled from policy,
        and $\gamma$ is the discount factor between $[0, 1]$.

        \begin{align}
        \mathbb{E}_{\tau \sim \pi_\theta} \Biggl[
        \sum_{t=0}^\infty \gamma^t A^{\pi_{OLD}}(s_t, a_t)
        \Biggr] &=
        \\
        \mathbb{E}_{\tau \sim \pi_\theta} \Biggl[
        \sum_{t=0}^\infty \gamma^t \Bigl(
        Q^{\pi_{OLD}}(s_t, a_t) - V^{\pi_{OLD}}(s_t)
        \Bigr)
        \Biggr] &=
        \\
        \mathbb{E}_{\tau \sim \pi_\theta} \Biggl[
        \sum_{t=0}^\infty \gamma^t \Bigl(
        r_t + V^{\pi_{OLD}}(s_{t+1}) - V^{\pi_{OLD}}(s_t)
        \Bigr)
        \Biggr] &=
        \\
        \mathbb{E}_{\tau \sim \pi_\theta} \Biggl[
        \sum_{t=0}^\infty \gamma^t \Bigl(
        r_t
        \Bigr)
        \Biggr]
        - \mathbb{E}_{\tau \sim \pi_\theta}
            \Biggl[V^{\pi_{OLD}}(s_0)\Biggr] &=
        J(\pi_\theta) - J(\pi_{\theta_{OLD}})
        \end{align}

        So,
        $$\max_\theta J(\pi_\theta) =
        \max_\theta \mathbb{E}_{\tau \sim \pi_\theta} \Biggl[
            \sum_{t=0}^\infty \gamma^t A^{\pi_{OLD}}(s_t, a_t)
        \Biggr]$$

        Define discounted-future state distribution,
        $$d^\pi(s) = (1 - \gamma) \sum_{t=0}^\infty \gamma^t P(s_t = s | \pi)$$

        Then,

        \begin{align}
        J(\pi_\theta) - J(\pi_{\theta_{OLD}})
        &= \mathbb{E}_{\tau \sim \pi_\theta} \Biggl[
        \sum_{t=0}^\infty \gamma^t A^{\pi_{OLD}}(s_t, a_t)
        \Biggr]
        \\
        &= \frac{1}{1 - \gamma}
        \mathbb{E}_{s \sim d^{\pi_\theta}, a \sim \pi_\theta} \Bigl[
        A^{\pi_{OLD}}(s, a)
        \Bigr]
        \end{align}

        Importance sampling $a$ from $\pi_{\theta_{OLD}}$,

        \begin{align}
        J(\pi_\theta) - J(\pi_{\theta_{OLD}})
        &= \frac{1}{1 - \gamma}
        \mathbb{E}_{s \sim d^{\pi_\theta}, a \sim \pi_\theta} \Bigl[
        A^{\pi_{OLD}}(s, a)
        \Bigr]
        \\
        &= \frac{1}{1 - \gamma}
        \mathbb{E}_{s \sim d^{\pi_\theta}, a \sim \pi_{\theta_{OLD}}} \Biggl[
        \frac{\pi_\theta(a|s)}{\pi_{\theta_{OLD}}(a|s)} A^{\pi_{OLD}}(s, a)
        \Biggr]
        \end{align}

        Then we assume $d^\pi_\theta(s)$ and  $d^\pi_{\theta_{OLD}}(s)$ are similar.
        The error we introduce to $J(\pi_\theta) - J(\pi_{\theta_{OLD}})$
        by this assumption is bound by the KL divergence between
        $\pi_\theta$ and $\pi_{\theta_{OLD}}$.
        [Constrained Policy Optimization](https://arxiv.org/abs/1705.10528)
        shows the proof of this. I haven't read it.


        \begin{align}
        J(\pi_\theta) - J(\pi_{\theta_{OLD}})
        &= \frac{1}{1 - \gamma}
        \mathop{\mathbb{E}}_{s \sim d^{\pi_\theta} \atop a \sim \pi_{\theta_{OLD}}} \Biggl[
        \frac{\pi_\theta(a|s)}{\pi_{\theta_{OLD}}(a|s)} A^{\pi_{OLD}}(s, a)
        \Biggr]
        \\
        &\approx \frac{1}{1 - \gamma}
        \mathop{\mathbb{E}}_{\textcolor{orange}{s \sim d^{\pi_{\theta_{OLD}}}}
        \atop a \sim \pi_{\theta_{OLD}}} \Biggl[
        \frac{\pi_\theta(a|s)}{\pi_{\theta_{OLD}}(a|s)} A^{\pi_{OLD}}(s, a)
        \Biggr]
        \\
        &= \frac{1}{1 - \gamma} \mathcal{L}^{CPI}
        \end{align}
        
        ref: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/rl/ppo/__init__.py#L34
        """
        ratio = torch.exp(log_pi - sampled_log_pi)
        
        # Clamping the grads of the ratio to avoid exploding gradients
        # ratio.register_hook(lambda grad: grad.clamp(max=1e4))
        # ratio.register_hook(lambda t: print(f'\n hook ratio: \n {t}')) # print grads
        
        clipped_ratio = torch.clamp(ratio, 1 - clip, 1 + clip)
        policy_reward  = torch.min(ratio * advantage, clipped_ratio * advantage)
        
        # Note that the logs of the probs (<1) are negative; so, we use -1 * policy_reward
        policy_loss = - policy_reward
        
        return policy_loss
    
    @staticmethod
    def clipped_value_loss(value: torch.Tensor, 
                           sampled_value: torch.Tensor, 
                           sampled_return: torch.Tensor, 
                           clip: float
                           ) -> torch.Tensor:
        
        """
        ## Clipped Value Function Loss

        Similarly we clip the value function update also.

        \begin{align}
        V^{\pi_\theta}_{CLIP}(s_t)
        &= clip\Bigl(V^{\pi_\theta}(s_t) - \hat{V_t}, -\epsilon, +\epsilon\Bigr)
        \\
        \mathcal{L}^{VF}(\theta)
        &= \frac{1}{2} \mathbb{E} \biggl[
        max\Bigl(\bigl(V^{\pi_\theta}(s_t) - R_t\bigr)^2,
            \bigl(V^{\pi_\theta}_{CLIP}(s_t) - R_t\bigr)^2\Bigr)
        \biggr]
        \end{align}

        Clipping makes sure the value function $V_\theta$ doesn't deviate
        significantly from $V_{\theta_{OLD}}$.
        
        ref: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/rl/ppo/__init__.py#L34

        """
        
        clipped_value = sampled_value + (value - sampled_value).clamp(min=-clip, max=clip)
        value_loss = torch.max((value - sampled_return) ** 2, (clipped_value - sampled_return) ** 2)
        
        # Note that this multiplies by 0.5 in here and the overall loss calculation below
        return 0.5 * value_loss
        
    def calc_losses(self, 
                    agent: int , 
                    terminal_length: int = 25
                    ) -> torch.Tensor:
        """Calculate the losses.

        Args:
            agent (int): the agent index.
            terminal_length (int, optional): _description_. Defaults to 25.

        Returns:
            Tensor: the loss tensor.
        """
        buffer = self.episode_buffer.get_transitions()
        
        # Original shape: (batch_size, num_agents, state_size)
        sampled_state = buffer['state'][:terminal_length, agent, :] # Shape: (batch_size, state_size)
        sampled_state_central = buffer['state_central'][:terminal_length + 1, agent, :] # Shape: (batch_size +1, state_size_central)
        sampled_msg_sent = buffer['msg_sent'][:terminal_length, agent, :] # Shape: (batch_size, msg_size)
        sampled_msg_rec = buffer['msg_rec'][:terminal_length, agent, :] # Shape: (batch_size, msg_size)
        sampled_action = buffer['action'][:terminal_length, agent] # Shape: (batch_size)
        sampled_log_prob = buffer['action_log_prob'][:terminal_length, agent] # Shape: (batch_size)
        sampled_reward = buffer['reward'][:terminal_length] # Shape: (batch_size)
        sampled_terminal = buffer['terminal'][:terminal_length] # Shape: (batch_size)
        
        sampled_terminal = torch.tensor(sampled_terminal, dtype=torch.bool)
        
        # Sample values; no need for gradients here
        with torch.no_grad():
            # Shape: (batch_size + 1)
            sampled_value = self.local_AC.getValue(sampled_state_central).reshape(-1)
        
        # Calculate advantages using GAE -> Shape: (batch_size + 1?)
        sampled_advantages = self.calculate_advantage_and_gae(sampled_reward, 
                                                              sampled_value, 
                                                              sampled_terminal
                                                              )
        
        # Notice that values has shape: (batch_size + 1), while all the others have shape: (batch_size)
        # sampled_value_without_last_obs = sampled_value[:-1]
        sampled_value = sampled_value[:-1]
        sampled_return = sampled_value + sampled_advantages # TD target
        sampled_normalized_advantage = self.normalize_advantage(sampled_advantages)
        
        # Needed for gradient calculations.
        action, log_prob, entropy_bonus = self.local_AC.action_prob_entropy(sampled_state, sampled_msg_rec)
        value = self.local_AC.getValue(sampled_state_central).reshape(-1)
        
        policy_loss = self.clipped_ppo_loss(log_prob, 
                                            sampled_log_prob, 
                                            sampled_normalized_advantage, 
                                            clip=0.1
                                            )
        
        # Check this for more details: 
        # https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/rl/ppo/experiment.py
        value = value[:-1]
        value_loss = self.clipped_value_loss(value, sampled_value, sampled_return, clip=0.1)
        
        #######################################################
        # TODO: What is this loss? msg gradients already calculated in policy loss?
        # msg_sent -> Shape: (self.buffer_capacity, self.msg_size)
        # comm_loss = msg_sent.mean() - 0.01 * msg_rec.mean()
        
        # policy_loss + c1*value_loss - c2*entropy
        # Note that c1=0.5 and c2=0.01 are hyperparameters.
        loss = (policy_loss
                + 0.5 * value_loss
                - 0.01 * entropy_bonus
                ).mean()
        
        logger.debug(f"\n policy_loss: {policy_loss}, \n value_loss: {value_loss}, \n loss: {loss}")
        
        return loss
        
    def update_network(self, loss: Tensor, msg, curr_comm):
        """Updates the global network with the loss, then clips the weights in the global network.
        
        Args:
            `loss`: The loss of the network
            
        Returns:
            None
        """
        
        # Zero the gradients of the global network.
        self.optimizer.zero_grad()
        
        # Backward pass: compute gradient of the loss with respect to each network parameter.
        # retain_graph=True for the first n-1 backward passes
        # policy_loss.backward(retain_graph=True)
        # value_loss.backward(retain_graph=True)
        # comm_loss.backward(retain_graph=True)
        
        # loss = policy_loss + value_loss + comm_loss
        loss.backward(retain_graph=True)
        
        # Update the grads in the global network using the gradients of the local networks.
        for local_param, global_param in zip(
                self.local_AC.parameters(),
                self.global_AC.parameters()):
            global_param._grad = local_param.grad
            
        # Clip gradients in the global network to prevent exploding gradients.
        torch.nn.utils.clip_grad_norm_(parameters=self.global_AC.parameters(), max_norm=10)
        
        logger.info(f"""--------------------------------
                    \ncomm_fn weight grads: {self.local_AC.comm_fn[0].weight.grad}
                    \nhidden_comm weight grads: self.local_AC.hidden_comm[0].weight.grad
                    \npolicy weight grads: {self.local_AC.policy[0].weight.grad}
                    \nvalue weight grads: {self.local_AC.value.weight.grad}
                    \nmsg grads: {msg.grad}
                    \ncurr_comm grads: {curr_comm.grad}
                    \n--------------------------------""")
        
        # RuntimeError: One of the differentiated Tensors appears to not have been used in the graph. 
        # Set allow_unused=True if this is the desired behavior.
        # params = torch.autograd.grad(loss, self.local_AC.parameters(), retain_graph=True, allow_unused=True)
        # logger.info(f"\nparams: {params}")
                          
        # Update the global network using the gradients above.
        # self.policy_optimizer.step()
        # self.value_optimizer.step()
        # self.comm_optimizer.step()
        self.optimizer.step()
        
        # self.local_AC.load_state_dict(self.global_AC.state_dict())
        # self.local_AC.save_checkpoint(model_name=self.model_path)
    
    def output_mess_to_input_mess(self, 
                                  message: Tensor, 
                                  episode_comm_maps: List[List[Union[float, int]]]
                                  ) -> Tensor :
        """__Summary__:
        - converts sent messages into received messages
        - if spread_messages
          - then agent0 has output [mess0_1, mess0_2], and agent1 and agent2 have inputs mess0_1 and mess0_2
          - the sent messages mess0_1 and mess0_2 are copied into input of agent1 and agent2
        - else if not spread_messages
          - then agent0 has output mess0, and agent1 and agent2 have inputs mess0_1 and mess0_2
          - the sent message mess0 is copied into input of agent1 and agent2
          
        Args:
            message (Tensor[Tensor[Union[float, int]]]): the communication message sent by other agents -> shape: (num_agents, message_size)
            episode_comm_maps (List[List[Union[float, int]]]): the communication map that states which agent sent a message to which other agent -> shape: (batch_size, num_agents, num_agents-1)
            
        Returns:
            curr_comm (Tensor[Tensor[Union[float, int]]]): The mapped messages 
        """
        # curr_comm = []
        # no_mess = np.zeros(self.message_size)

        # if self.spread_messages:
        #     # for agent_state in states:
        #     for j, agent_state in enumerate(episode_comm_maps):
        #         curr_agent_comm = []
        #         for neighbor in agent_state:
        #             if neighbor != -1:
        #                 # print("message from ", neighbor, "to", j)
        #                 # TODO this is incorrect, it will copy entire message for all agents and not the specific one
        #                 curr_agent_comm.extend(message[neighbor])
        #             else:
        #                 curr_agent_comm.extend(no_mess)
        #         curr_comm.append(curr_agent_comm)
        # else:
        #     # for agent_state in states:
        #     for j, agent_state in enumerate(episode_comm_maps):
        #         curr_agent_comm = []
        #         # print(agent_state)
        #         for neighbor in agent_state:
        #             if neighbor != -1:
        #                 # print("message from ", neighbor, "to", j)
        #                 curr_agent_comm.extend(message[neighbor])
        #             else:
        #                 curr_agent_comm.extend(no_mess)
        #         curr_comm.append(curr_agent_comm)

        # logger.debug(f"""output_mess_to_input_mess -> 
        #              \nsending message: \n\t{message}
        #              \ncomm_map: \n\t{episode_comm_maps}
        #              \ncurr_comm: \n\t{curr_comm}
        #              """)

        # return curr_comm
                
        curr_comm = []
        
        # msg = [[msg_size], [msg_size], ..., [msg_size]]
        for i in range(self.number_of_agents):
            cat_tns = torch.cat((message[:i].reshape(-1), message[i+1:].reshape(-1)), dim=0)
                        
            curr_comm.append(cat_tns) 
        
        curr_comm = torch.stack(curr_comm, dim=0)
        logger.debug(f"\n output_mess_to_input_mess: \n curr_comm: {curr_comm}")
                
        return curr_comm

    

