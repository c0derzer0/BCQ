from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Module):
    def __init__(self, frames, num_actions):
        super(Conv, self).__init__()
        self.c1 = nn.Conv2d(frames, 32, kernel_size=8, stride=4)
        self.c2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.c3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.q1 = nn.Linear(3136, 512)
        self.q2 = nn.Linear(512, num_actions)

        self.i1 = nn.Linear(3136, 512)
        self.i2 = nn.Linear(512, num_actions)


    def forward(self, state):
        c = F.relu(self.c1(state))
        c = F.relu(self.c2(c))
        c = F.relu(self.c3(c))

        q = F.relu(self.q1(c.reshape(-1, 3136)))
        i = F.relu(self.i1(c.reshape(-1, 3136)))
        i = self.i2(i)
        return self.q2(q), F.log_softmax(i, dim=1), i


class DBCQ(object):
    def __init__(self, parameters, env_properties, device):
        self.device = device

        self.critic = Conv(4, env_properties["num_actions"]).to(self.device)
        self.critic_target = deepcopy(self.critic)
        self.critic_optimizer = getattr(torch.optim, parameters["optimizer"])(
            self.critic.parameters(), **parameters["optimizer_parameters"]
        )

        # Parameters for train()
        self.discount = parameters["discount"]

        # Select target update rule
        # copy: copy full target network every "target_update_frequency" iterations
        # polyak: update every timestep with proportion tau
        self.maybe_update_target = self.polyak_target_update if parameters["polyak_target_update"] \
            else self.copy_target_update
        self.target_update_frequency = parameters["target_update_frequency"] \
            / parameters["update_frequency"]
        self.tau = parameters["tau"]

        # Parameters for exploration + Compute linear decay for epsilon
        self.initial_epsilon = parameters["initial_epsilon"]
        self.end_epsilon = parameters["end_epsilon"]
        self.slope = (self.end_epsilon - self.initial_epsilon) \
            / parameters["epsilon_decay_period"] * parameters["update_frequency"]

        # Parameters for evaluation
        self.state_shape = (-1, 4, 84, 84) if env_properties["atari"] else (-1, env_properties["state_dim"])
        self.evaluation_epsilon = parameters["evaluation_epsilon"]
        self.num_actions = env_properties["num_actions"]

        self.threshold = 0.3

        # Number of training iterations
        self.iterations = 0

    def select_action(self, state, eval=False):
        eps = self.evaluation_epsilon if eval \
            else max(self.slope * self.iterations + self.initial_epsilon, self.end_epsilon)

        # Select action according to policy with probability (1-eps)
        # otherwise, select random action
        if np.random.uniform(0,1) > eps:
            with torch.no_grad():
                state = torch.FloatTensor(state).reshape(1, 4, 84, 84).to(self.device)
                q, imt, i = self.critic(state)
                imt = imt.exp()
                imt = (imt/imt.max(1, keepdim=True)[0] > self.threshold).float()
                return int((imt * q + (1 - imt) * -1e6).argmax(1))
        else:
            return np.random.randint(self.num_actions)

    def get_q_values(self, state):

        with torch.no_grad():
            state = torch.FloatTensor(state).reshape(1, 4, 84, 84).to(
                self.device)
            q, imt, i = self.critic(state)
            imt = imt.exp()
            imt = (imt / imt.max(1, keepdim=True)[
                0] > self.threshold).float()
        return q, imt, i


    def train(self, replay_buffer):
        # Sample replay buffer
        state, action, next_state, reward, done = replay_buffer.sample\
            (replay_buffer.size)

        # Compute the target Q value
        with torch.no_grad():
            q, imt, i = self.critic(next_state)
            imt = imt.exp()
            imt = (imt/imt.max(1, keepdim=True)[0] > self.threshold).float()
            next_action = (imt * q + (1 - imt) * -1e6).argmax(1, keepdim=True)

            q, imt, i = self.critic_target(next_state)
            target_Q = reward + done * self.discount * q.gather(1, next_action).reshape(-1, 1)

        # Get current Q estimate
        current_Q, imt, i = self.critic(state)
        current_Q = current_Q.gather(1, action)

        # Compute critic loss
        q_loss = F.smooth_l1_loss(current_Q, target_Q)
        i_loss = F.nll_loss(imt, action.reshape(-1))

        critic_loss = q_loss + i_loss + 1e-2 * i.pow(2).mean()

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update target network by polyak or full copy every X iterations.
        self.iterations += 1
        self.maybe_update_target()


    def polyak_target_update(self):
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
           target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


    def copy_target_update(self):
        if self.iterations % self.target_update_frequency == 0:
             self.critic_target.load_state_dict(self.critic.state_dict())
