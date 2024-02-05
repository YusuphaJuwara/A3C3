# @title Imports

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# from python 3.9 below, they don't support list, tuple, etc; so, import them from typing instead
from typing import Union, List, Tuple

# Logging
import logging
from logger import setLogger

# Change the logging level ...
logger = setLogger(name="Actor-Critic", set_level=logging.INFO)

class BaseNetwork(nn.Sequential):
    """
    Linear layer with an activation function stack
    
    Args:
        in_features (int): number of input features to be used for the linear layer.
        out_features (int): number of output features to be used for the linear layer.
        activation (Union[nn.ReLU, nn.ELU, nn.Sigmoid,nn.Softmax, nn.Tanh]): activation function to be used.
    
    Attributes:
        nn.Sequential: nn.Sequential(
            nn.Linear(in_features, out_features),
            activation(dim=-1) if activation == nn.Softmax else activation()
        )
    """
    def __init__(self, 
                 in_features:int, 
                 out_features:int, 
                 activation: Union[nn.ReLU, nn.ELU, nn.Sigmoid,nn.Softmax, nn.Tanh] = nn.ReLU
                 ):
        super(BaseNetwork, self).__init__(
            nn.Linear(in_features, out_features),
            activation(dim=-1) if activation == nn.Softmax else activation()
        )
        
        
# @title Actor-Critic Network
class AC_Network(nn.Module):
    
    """_summary_: Actor-Critic Network.
    
    Args:
        s_size (List[int]): the size of the (partial) observation space. E.g., [6].
        s_size_central (List[int]): the size of the central observation space. E.g., [8].
        number_of_agents (int): the number of agents in the environment.
        a_size (int): the size of the action space. E.g., 5.
        comm_size_input (int): the communication size of the input. E.g., 20.
        comm_size_output (int): the communication size of the output. E.g., 20.
        critic_action (bool): whether to use critic for action.
        critic_comm (bool): whether to use critic for communication.
        paramSearch (Tuple[int, str, int, int]): the search space for the hyperparameters.
        
    Attributes:
        self.obs_hidden (nn.Sequential): the hidden layer for the policy network.
        self.central_obs_hidden (nn.Sequential): the hidden layer for the central value network.
        self.value (nn.Linear): the central value network.
        self.policy (nn.Sequential): the policy network.
        self.hidden_comm (nn.Sequential): the hidden layer of the communication space.
        self.comm_fn (Union[nn.Sequential, nn.Linear]): the communication network.
        self.cat_dist (Categorical): the Categorical distribution for discrete action spaces.
        
    Example:
    ```python
        net = AC_Network(s_size=[6], 
                        s_size_central=[8], 
                        number_of_agents=2, 
                        a_size=5,
                        comm_size_input=20,
                        comm_size_output=20,
                        critic_action=False,
                        critic_comm=False
                        paramSearch=[40,"relu",80,40]
                        )
    ```
    """
    
    def __init__(self,
                 s_size: List[int] = [6],
                 s_size_central: List[int] = [8],
                 number_of_agents: int = 2,
                 a_size: int = 5,
                 comm_size_input: int = 2,
                 comm_size_output: int = 2,
                 critic_action: bool=False,
                 critic_comm: bool=False,
                 paramSearch: Tuple[int, str, int, int] = [40,"relu",80,40]
                 ):
        """
        s_size:self.agent_observation_space = [2 + 2 * number_of_agents] -> list[int]
        s_size_central: self.central_observation_space = [2 * number_of_agents + 2 * number_of_agents] -> list[int]
        a_size: self.agent_action_space = self.max_actions = 5 -> int

        # A tuple corresponding to the min and max possible rewards
        self.reward_range = [0, self.number_of_agents]

        paramSearch = (40,"relu",80,40) = [size_of_comm_HL, activ_fn, size_of_actor_and_critic_HL1, size_of_actor_and_critic_HL2, ...]"
        """
        super(AC_Network, self).__init__()
        self.s_size = s_size
        self.s_size_central = s_size_central
        self.number_of_agents = number_of_agents
        self.a_size = a_size
        self.comm_size_input = comm_size_input
        self.comm_size_output = comm_size_output
        self.critic_action = critic_action
        self.critic_comm = critic_comm
        self.paramSearch = paramSearch
                
        torch.autograd.set_detect_anomaly(True)

        # Get the correct central input size -> type: list[int]
        if critic_action and critic_comm:
            central_input_size = [s_size_central[0] + (number_of_agents - 1)*a_size + comm_size_input]
        elif critic_comm:
            central_input_size = [s_size_central[0] + comm_size_input]
        elif critic_action:
            central_input_size = [s_size_central[0] + (number_of_agents - 1) * a_size]
        else:
            central_input_size = s_size_central
            
        logger.debug(f"central_input_size: {central_input_size}")

        # Activation function
        if paramSearch[1] == "elu":
            activation = nn.ELU
        elif paramSearch[1] == "sig":
            activation = nn.Sigmoid
        elif paramSearch[1] == "soft":
            activation = nn.Softmax
        else:
            activation = nn.ReLU

        obs_hidden = []
        central_hidden = []

        s = s_size[0] + comm_size_input # type: int
        c_s = central_input_size[0]     # type: int
        for idx, layer_size in enumerate(paramSearch[2:]):

            obs_hidden.append(
                # takes both state and msg as input
                BaseNetwork(in_features=s, out_features=layer_size, activation=activation)
                )

            central_hidden.append(
                BaseNetwork(in_features=c_s, out_features=layer_size, activation=activation)
                )

            # Same except in the beginning
            s = layer_size
            c_s = layer_size

        self.obs_hidden = nn.Sequential(*obs_hidden)
        self.central_hidden = nn.Sequential(*central_hidden)

        # Takes the last central state as input
        self.value = nn.Linear(in_features=c_s, out_features=1)
        
        # Takes the state as input and outputs action probabilities of size `a_size`
        self.policy = BaseNetwork(in_features=s, out_features=a_size, activation=nn.Softmax)

        # only state as input
        self.hidden_comm = BaseNetwork(in_features=s_size[0], out_features=paramSearch[0], activation=activation)

        if comm_size_output != 0:
            self.comm_fn = BaseNetwork(in_features=paramSearch[0], out_features=comm_size_output, activation=nn.Tanh)
        else:
            self.comm_fn = nn.Linear(in_features=paramSearch[0], out_features=comm_size_output)
            
        # self.cat_dist(self.policy(obs)) -> Discreate distribution
        # takes as input either probs or logits, but not both
        self.cat_dist = torch.distributions.Categorical
            
        # self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        """The initial weights of the network. Inplace operation.

        Args:
            m (nn.Module): the module to be initialized with the weights.
            
        Returns:
            None -> nn.Module: the initialized module.
            
        Example:
        ```python
        >>> self.apply(self._weights_init)
        ```
        """
        if isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.001)
            nn.init.constant_(m.bias, 0)
                
    def forward(self, 
                obs: torch.Tensor, 
                OBS: torch.Tensor, 
                msg: torch.Tensor,
                ):
        """_summary_

        Args:
            obs (Tensor[Tensor[int, float]]): the (partial) observation of each agent.
            OBS (Tensor[Tensor[int, float]]): the centralized observation for all agents in the Worker.
            msg (Tensor[Tensor[FloatTensor]]): the message of each agent to be sent to the other agents in same Worker.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: policy, value, msg_out
            
        Example:
        ```python
        >>> import torch
        >>> obs = torch.tensor([[4, 2, 11, 1, 2, 12], [9, 12, 11, 1, 2, 12]])
        >>> OBS = torch.tensor([[4, 2, 9, 12, 11, 1, 2, 12], [4, 2, 9, 12, 11, 1, 2, 12]])
        >>> msg = torch.tensor([[-0.9992495, 0.19028412], [-0.99989754, -0.22812036]])
        >>> policy, value, msg_out = self.forward(obs, OBS, msg)
        ```
        """
        
        policy = self.getPolicy(obs, msg)
        value = self.getValue(OBS)
        msg_out = self.getMessage(obs)

        return policy, value, msg_out
    
    def getPolicy(self, 
                  obs: torch.Tensor , 
                  msg: torch.Tensor
                  ) -> torch.Tensor:
        """Policy part: Takes the state and msg as inputs and outputs action probabilities of size `a_size`

        Args:
            obs (Tensor[Tensor[int, float]]): the (partial) observation of each agent.
            msg (Tensor[Tensor[FloatTensor]]): the message of each agent to be sent to the other agents in same Worker.

        Returns:
            torch.Tensor: the action probabilities of each agent.
            
        Example:
        ```python
        >>> import torch
        >>> obs = torch.tensor([[4, 2, 11, 1, 2, 12], [9, 12, 11, 1, 2, 12]])
        >>> msg = torch.tensor([[-0.9992495, 0.19028412], [-0.99989754, -0.22812036]])
        >>> policy = self.getPolicy(obs, msg)
        ```
        """
        
        # Can be easily used to get the grad of the msg w.r.t. the policy
        self.msg_tns = torch.tensor(msg, dtype=torch.float, requires_grad=True)
        # self.msg_tns = torch.zeros_like(msg, dtype=torch.float, requires_grad=True)
                
        obs_with_msg = torch.cat((obs, self.msg_tns), dim=1)
        
        obs_hidden = self.obs_hidden(obs_with_msg)
        policy = self.policy(obs_hidden)

        return policy
    
    def getValue(self, OBS: torch.Tensor) -> torch.Tensor:
        """Centralized Value network

        Args:
            OBS (Tensor[Tensor[int]]): The centralized observation for all agents in the Worker.

        Returns:
            torch.Tensor (Tensor[float]): The centralized value for all agents in the Worker.
            
        Example:
        ```python
        >>> import torch
        >>> OBS = torch.tensor([[4, 2, 9, 12, 11, 1, 2, 12], [4, 2, 9, 12, 11, 1, 2, 12]])
        >>> value = self.getValue(OBS)
        ```
        """
        # Centralized Value network
        central_hidden = self.central_hidden(OBS)
        value = self.value(central_hidden)
        
        return value
    
    def getMessage(self, obs: torch.Tensor) -> torch.Tensor:
        """The `Communication Message` part.

        Args:
            obs (Tensor[Tensor[Union[float, int]]]): The (partial) observation of each agent

        Returns:
            torch.Tensor: The message of each agent to be sent to the other agents in same Worker.
            
        Example:
        ```python
        >>> import torch
        >>> obs = torch.tensor([[4, 2, 11, 1, 2, 12], [9, 12, 11, 1, 2, 12]])
        >>> msg_out = self.getMessage(obs)
        ```
        """
        msg_hidden = self.hidden_comm(obs)
        msg_out = self.comm_fn(msg_hidden)

        return msg_out
    
    def action_prob_entropy(self, 
                obs: torch.Tensor, 
                curr_comm: torch.Tensor,
                eval_model: bool = False
                ) -> List[int]:
        """Sample an action for each agent from the action distrbutions
            
        Args:
            obs (Tensor[Tensor[Union[float, int]]]): The (partial) observation of each agent
            curr_comm (Tensor[Tensor[Union[float, int]]]): The received message from other agents
            eval_mode (bool): Whether the model is in evaluation mode (True) or not (False).
        
        Returns: 
            actions (Tensor[int]): The sampled actions for each agent.
            log_probs (Tensor[float]): The log probabilities of the sampled actions.
            entropy (Tensor[float]): The entropy of the action distribution.
            
        Example:
        ```python
        >>> import torch
        >>> obs = torch.tensor([[4, 2, 11, 1, 2, 12], [9, 12, 11, 1, 2, 12]])
        >>> curr_comm = torch.tensor([[-0.9992495, 0.19028412], [-0.99989754, -0.22812036]])
        >>> actions, log_probs, entropy = self.action_prob_entropy(obs, curr_comm)
        ```
        """
                
        # Get the action probs from the policy network.
        # Shape: (batch_size (or num_agents), action_size=5)
        actions_probs = self.getPolicy(obs=obs, msg=curr_comm)
        
        logger.debug(f"action_prob_entropy: \n action_probs: {actions_probs}")
        
        # During evaluation, you might want to greedily choose actions with the maximum probability
        if eval_model:
            # Shape: (batch_size (or num_agents)), e.g., tensor([4, 4, 4])
            actions = torch.argmax(actions_probs, dim=1)
            return actions, None, None
        
        action_distribution = self.cat_dist(probs=actions_probs)
        
        # Sample an actions for each batch (agent)
        # Shape: (batch_size (or num_agents)), e.g., tensor([1, 1, 3])
        actions = action_distribution.sample()

        # Get log probabilities of the sampled actions
        # Shape: (batch_size (or num_agents)), 
        # e.g., tensor([-1.6094, -1.6094, -1.6094], grad_fn=<SqueezeBackward1>)
        log_probs = action_distribution.log_prob(actions)

        # Get entropy of the distribution (optional but can be useful for exploration)
        # Shape: (batch_size (or num_agents)), 
        # e.g., tensor([1.6094, 1.6094, 1.6094], grad_fn=<NegBackward0>)
        entropy = action_distribution.entropy()
        # dist_entropy = distribution.entropy().sum(1, keepdim=True)
        
        logger.debug(f"""action_prob_entropy:
                    \n action_distribution: {action_distribution}
                    \n actions: {actions}
                    \n log_probs: {log_probs}
                    \n entropy: {entropy}
                    """)
        
        return actions, log_probs, entropy
    
    def save_checkpoint(self, model_name='./models/actor_critic_1.pkl'):
        """Saves the model checkpoint(s) to the specified path

        Args:
            model_name (str, optional): The path to save the model. Defaults to './models/actor_critic_1.pkl'
        """
        torch.save(self.state_dict(), model_name)
        logger.info(f"\nCheckpoint saved!! in {model_name}\n")

    def load_checkpoint(self, model_name='./models/actor_critic_1.pkl'):
        """Loads the model checkpoint(s) from the specified path
            
        Args:
            model_name (str, optional): The path to load the model. Defaults to './models/actor_critic_1.pkl'
        """
        self.load_state_dict(torch.load(model_name))
        logger.info(f"\nCheckpoint loaded from {model_name}!!\n")

if __name__ == "__main__":
    
    paramSearch = [40,"relu",80,40]
    
    # obs: List[List[int]] = [[4, 2, 11, 1, 2, 12], [9, 12, 11, 1, 2, 12]]
    # OBS: List[List[int]] = [[4, 2, 9, 12, 11, 1, 2, 12], [4, 2, 9, 12, 11, 1, 2, 12]]
    # msg: List[Union[float, int]] = [[-0.9992495, 0.19028412], [-0.99989754, -0.22812036]]
    
    from gym_env import GymNav
    import torch
    number_of_agents = 3
    comm_size_input = 2
    env = GymNav(number_of_agents=number_of_agents)
    state_size = env.agent_observation_space
    s_size_central = env.central_observation_space
    action_size = env.agent_action_space
    reward_range = env.reward_range
    env.close()
    
    logger.info(f"""\nState size: {state_size}
                 \ns_size_central: {s_size_central}
                 \naction_size: {action_size}
                 \nreward_range: {reward_range}\n
                 """)
    
    from replay_buffer import ReplayBuffer
    buffer = ReplayBuffer(num_agents=number_of_agents,
                          buffer_capacity=256,
                          state_size=state_size[0],
                          state_size_central=s_size_central[0],
                          action_size=action_size,
                          msg_size=comm_size_input)
    
    obs, info = env.reset()
    obs = torch.tensor(obs, dtype=torch.float)
    OBS = info["state_central"]
    OBS = torch.tensor(OBS, dtype=torch.float)
    msg = torch.rand(size=(number_of_agents, comm_size_input))
    
    logger.info(f"obs: {obs}\OBS: {OBS}")
    
    # global_agents = AC_Network()
    global_agents = AC_Network( s_size=state_size,
                                s_size_central=s_size_central,
                                number_of_agents=number_of_agents,
                                a_size=action_size,
                                comm_size_input=comm_size_input,
                                comm_size_output=comm_size_input,
                                critic_action=False,
                                critic_comm=False,
                                paramSearch=paramSearch
                                )
            
    policy, value, msg_out = global_agents.forward(obs=obs, 
                                                    OBS=OBS, 
                                                    msg=msg
                                                    )

    logger.info(f"""
                 \n policy: {policy}
                 \n value: {value}
                 \n msg_out: {msg_out}
                 """)
    
    actions, log_probs, entropy = global_agents.action_prob_entropy(obs=obs, curr_comm=msg, eval_model=False)
    
    logger.info(f"\nactions: {actions}\nlog_probs: {log_probs}\nentropy: {entropy}\n")
