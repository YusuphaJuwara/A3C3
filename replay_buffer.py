# @title Replay Buffer

import numpy as np
import torch
# from np.distributions import Beta
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

class ReplayBuffer:
    '''Replay buffer for holding transitions of all agents in the environment.
    
    Args:
        num_agents (int, optional): The number of agents in the environment. Defaults to 2.
        buffer_capacity (int, optional): The capacity of the replay buffer. Defaults to 256.
        state_size (int, optional): The size of the state. Defaults to 8.
        state_size_central (int, optional): The size of the state central. Defaults to 12.
        action_size (int, optional): The size of the action. Defaults to 5.
        msg_size (int, optional): The size of the message. Defaults to 20.
        
    Attributes:
        buffer_capacity (int): The capacity of the replay buffer.
        counter (int): The number of transitions stored in the buffer.
        buffer (np.array): The replay buffer.
        transition_dtype (np.dtype): The data type of the transitions.
        transition_shape (tuple): The shape of the transitions.
        transition_shape_central (tuple): The shape of the transitions central.
        batch_size (int): The size of the batch.
        batch_size_central (int): The size of the batch central.
        
    Methods:
        clear_buffer(self) -> None
        add_transition(self, state: np.ndarray, state_central: np.ndarray, msg_sent: np.ndarray, msg_recv: np.ndarray, action: np.ndarray, reward: np.ndarray, done: np.ndarray) -> None
        sample_batch(self) -> tuple of (batch number of observations in the buffer, batch of observations, batch of actions, batch of rewards, batch of dones)
        sample_batch_central(self) -> tuple of (batch number of observations in the buffer, batch of observations central)
        get_transitions(self) -> tuple of (state, state_central, msg_sent, msg_rec, action, action_log_prob, reward, terminal)
        get_transitions_central(self) -> tuple of (state_central, msg_sent, msg_rec, action, action_log_prob, reward, terminal)
        store_last_observation(self, state_central) -> tuple of (state_central)
        store(self, state, state_central, msg_sent, msg_rec, action, action_log_prob, reward, terminal) -> bool
    '''
    def __init__(self, 
                 num_agents: int = 2, 
                 buffer_capacity: int = 256, 
                 state_size: int = 8, 
                 state_size_central: int = 12, 
                 action_size: int = 5,
                 msg_size: int = 20
                 ) -> None:
        self.num_agents = num_agents
        self.buffer_capacity = buffer_capacity
        self.state_size = state_size
        self.state_size_central = state_size_central
        self.action_size = action_size
        self.msg_size = msg_size
                
        self.buffer_capacity = max(self.buffer_capacity, 1) # at least 1 elem
        self.clear_buffer()

    def clear_buffer(self) -> None:
        """_summary_: clear/initialize the replay buffer.
        """
        self.counter = 0
        # self.buffer = np.empty(self.buffer_capacity, dtype=self.transition_dtype)
        
        self.buffer = {
            'state': torch.empty((self.buffer_capacity, self.num_agents, self.state_size), dtype=torch.float32),
            'state_central': torch.empty((self.buffer_capacity + 1, self.num_agents, self.state_size_central), dtype=torch.float32),
            'msg_sent': torch.empty((self.buffer_capacity, self.num_agents, self.msg_size), dtype=torch.float32),
            'msg_rec': torch.empty((self.buffer_capacity, self.num_agents, self.msg_size * (self.num_agents -1)), dtype=torch.float32),
            'action': torch.empty((self.buffer_capacity, self.num_agents), dtype=torch.float),
            'action_log_prob': torch.empty((self.buffer_capacity, self.num_agents), dtype=torch.float),
            'reward': torch.empty(self.buffer_capacity, dtype=torch.float32),
            'terminal': np.empty(self.buffer_capacity, dtype=np.bool_)
        }
        
        self.buffer['terminal'].fill(False)

    def store(self, state, 
            state_central, 
            msg_sent, msg_rec, 
            action, action_log_prob,
            reward, 
            terminal,
            ) -> bool:
        '''Store transitions to the state buffer. 
        Note that these are not batch levels, but for all agents at onces.
        E.g., state = tensor([[2, 3], [4, 5], [6, 7]]) -> means for three agents; not one agent's batch...
        
        Args:
            state (Tensor[Tensor[float|int]]]): The (partial) observation state.
            state_central (Tensor[Tensor[float|int]]]): The centralized observation state.
            msg_sent (Tensor[Tensor[float]]]): The message sent by each agent.
            msg_rec (Tensor[Tensor[float]]]): The message received by each agent.
            action (Tensor[int]): The action taken by each agent.
            action_log_prog (Tensor[float]): The log probabilities of each action.
            reward (Tensor[float]): The reward received (by all the agents).
            terminal (ndarray[bool_]): The terminal state.
            
        Returns:
            bool: Whether the replay buffer is full or not.
            
        Example:
        ```python
        >>> state = tensor([[2, 3], [4, 5], [6, 7]]) 
        >>> state_central = tensor([[2, 3, 5], [4, 5, 2], [6, 7, 3]])
        >>> msg_sent = tensor([[2.4333, 1.0932], [2.1432, 6.1234], [0.23082, -1.0932]])
        >>> msg_rec = tensor([[2.1432, 6.1234, 0.23082, -1.0932], [2.4333, 1.0932, 0.23082, -1.0932], [2.1432, 6.1234, 0.23082, -1.0932]])
        >>> action = tensor([2, 3])
        >>> action_log_prob = tensor([log 0.8 = -0.09691, log 0.2 = -0.69897])
        >>> reward = tensor([0.33342])
        >>> terminal = [False]
        >>> replay_buffer.store(state, state_central, msg_sent, msg_rec, action, action_log_prob, reward, terminal)
        ```
        '''
                
        state = torch.Tensor(state)
        state_central = torch.Tensor(state_central)
        
        self.buffer['state'][self.counter] = state
            
        self.buffer['state_central'][self.counter] = state_central
        self.buffer['msg_sent'][self.counter] = msg_sent
        self.buffer['msg_rec'][self.counter] = msg_rec
        self.buffer['action'][self.counter] = action
        self.buffer['action_log_prob'][self.counter] = action_log_prob
        self.buffer['reward'][self.counter] = reward
        self.buffer['terminal'][self.counter] = terminal

        #self.buffer[self.counter] = transition
        self.counter += 1
        is_full = (self.buffer_capacity - self.counter) == 0

        self.counter %= self.buffer_capacity
        return is_full
    
    def store_last_observation(self, state_central: torch.Tensor) -> None:
        """_summary_: store the last centralized observation in the buffer.

        Args:
            state_central (Tensor[Tensor[float|int]]): The centralized observation state.
            
        Example:
        ```python
        >>> state_central = tensor([[2, 3, 5], [4, 5, 2], [6, 7, 3]])
        >>> replay_buffer.store_last_observation(state_central)
        ```
        """
        # Note that observation are (batch_size + 1 ...)
        self.buffer['state_central'][self.counter + 1] = state_central

    def get_transitions(self):
        '''Get all the transitions of size buffer_capacity'''
        return self.buffer

    def sample_transitions(self, batch_size):
        """
        - SubsetRandomSampler -> Samples elements randomly from a given list of indices, without replacement
        - BatchSampler -> Wraps another sampler to yield a mini-batch of indices.
        
        ```python
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        ```
        
        Args:
            batch_size (int): the number of transitions to be sampled.

        Returns:
           Indices (List[int]) : Indices of transitions to be sampled from the replay buffer.
           
        Example:
        >>> indices = replay_buffer.sample_transitions(batch_size=25)
        """
        indices = BatchSampler(SubsetRandomSampler(range(self.buffer_capacity)), batch_size, False)
        return indices