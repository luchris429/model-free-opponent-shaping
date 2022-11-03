import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MemoryMFOS:
    def __init__(self):
        self.actions_traj = []
        self.states_traj = []
        self.logprobs_traj = []
        self.rewards = []

    def clear_memory(self):
        del self.actions_traj[:]
        del self.states_traj[:]
        del self.logprobs_traj[:]
        del self.rewards[:]


class ActorCriticMFOS(nn.Module):
    def __init__(self, input_shape, action_dim, n_out_channels, batch_size):
        super(ActorCriticMFOS, self).__init__()
        self.batch_size = batch_size
        self.n_out_channels = n_out_channels
        self.space = n_out_channels

        self.conv_a_0 = nn.Conv2d(input_shape[0], n_out_channels, kernel_size=3, stride=1, padding="same", padding_mode="circular")
        self.conv_a_1 = nn.Conv2d(n_out_channels, n_out_channels, kernel_size=3, stride=1, padding="same", padding_mode="circular")
        self.linear_a_0 = nn.Linear(n_out_channels * input_shape[1] * input_shape[2], self.space)

        self.GRU_a = nn.GRU(input_size=self.space, hidden_size=self.space)
        self.linear_a = nn.Linear(self.space, action_dim)

        self.conv_v_0 = nn.Conv2d(input_shape[0], n_out_channels, kernel_size=3, stride=1, padding="same", padding_mode="circular")
        self.conv_v_1 = nn.Conv2d(n_out_channels, n_out_channels, kernel_size=3, stride=1, padding="same", padding_mode="circular")
        self.linear_v_0 = nn.Linear(n_out_channels * input_shape[1] * input_shape[2], self.space)

        self.GRU_v = nn.GRU(input_size=self.space, hidden_size=self.space)
        self.linear_v = nn.Linear(self.space, 1)

        self.conv_t_0 = nn.Conv2d(input_shape[0], n_out_channels, kernel_size=3, stride=1, padding="same", padding_mode="circular")
        self.conv_t_1 = nn.Conv2d(n_out_channels, n_out_channels, kernel_size=3, stride=1, padding="same", padding_mode="circular")
        self.linear_t_0 = nn.Linear(n_out_channels * input_shape[1] * input_shape[2], self.space)
        self.GRU_t = nn.GRU(input_size=self.space, hidden_size=self.space)
        self.linear_t = nn.Linear(self.space, self.space)

        self.reset(None, outer=True)

    def reset(self, memory, outer=False):
        if not outer:
            state_tbs = torch.stack(self.state_traj_bs, dim=0)
            memory.states_traj.append(state_tbs)
            memory.actions_traj.append(torch.stack(self.action_traj_b, dim=0))
            memory.logprobs_traj.append(torch.stack(self.logprob_traj_b, dim=0))
            state_Bs = state_tbs.flatten(end_dim=1)
            x = self.conv_t_0(state_Bs)
            x = F.relu(x)
            x = self.conv_t_1(x)
            x = F.relu(x)
            x = torch.flatten(x, start_dim=1)
            x = self.linear_t_0(x)
            x = F.relu(x)
            x = x.view(len(self.state_traj_bs), self.batch_size, self.space)
            x, _ = self.GRU_t(x)
            x = x[-1]
            x = x.mean(0, keepdim=True).repeat(self.batch_size, 1)
            # ACTIVATION HERE?
            x = self.linear_t(x)
            x = torch.sigmoid(x)
            self.th_bh = x
        else:
            self.th_bh = torch.ones(self.batch_size, self.space).to(device)

        self.ah_obh = torch.zeros(1, self.batch_size, self.space).to(device)
        self.vh_obh = torch.zeros(1, self.batch_size, self.space).to(device)

        self.state_traj_bs = []
        self.action_traj_b = []
        self.logprob_traj_b = []

    def forward(self):
        raise NotImplementedError

    def forward_a(self, state_bs):
        x = self.conv_a_0(state_bs)
        x = F.relu(x)
        x = self.conv_a_1(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.linear_a_0(x)
        x = F.relu(x)
        x, self.ah_obh = self.GRU_a(x.unsqueeze(0), self.ah_obh)
        x = x.squeeze(0)
        x = F.relu(x)
        x = self.th_bh * x

        return F.softmax(self.linear_a(x), dim=-1)

    def act(self, state_bs):
        action_probs_ba = self.forward_a(state_bs)
        dist = Categorical(action_probs_ba)
        action_b = dist.sample()

        self.state_traj_bs.append(state_bs)
        self.action_traj_b.append(action_b)
        self.logprob_traj_b.append(dist.log_prob(action_b))
        return action_b

    def evaluate(self, state_Ttbs, action_Ttb):

        state_Bs = state_Ttbs.transpose(0, 1).flatten(end_dim=2)

        x = self.conv_t_0(state_Bs)
        x = F.relu(x)
        x = self.conv_t_1(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.linear_t_0(x)
        x = F.relu(x)
        x = x.view(state_Ttbs.size(1), state_Ttbs.size(0) * self.batch_size, self.space)
        x, _ = self.GRU_t(x)  # tBh
        x = x[-1]  # Bh
        x = x.view(state_Ttbs.size(0), self.batch_size, self.space)  # Tbh
        x = x.mean(1, keepdim=True).repeat(1, self.batch_size, 1)  # Tbh
        # ACTIVATION HERE?
        x = self.linear_t(x)  # Tbh
        x = torch.sigmoid(x)
        th_Tbh = torch.cat((torch.ones(1, self.batch_size, self.space).to(device), x[:-1]), dim=0)

        x = self.conv_a_0(state_Bs)
        x = F.relu(x)
        x = self.conv_a_1(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.linear_a_0(x)
        x = F.relu(x)
        x = x.view(state_Ttbs.size(1), state_Ttbs.size(0) * self.batch_size, self.space)  # tBh
        x, _ = self.GRU_a(x)  # tBh
        x = x.view(state_Ttbs.size(1), state_Ttbs.size(0), self.batch_size, self.space)  # tTbh
        x = x.transpose(0, 1)  # Ttbh
        x = F.relu(x)
        x = th_Tbh.unsqueeze(1) * x
        action_probs_Ttba = F.softmax(self.linear_a(x), dim=-1)
        dist = Categorical(action_probs_Ttba)
        action_logprobs = dist.log_prob(action_Ttb)
        dist_entropy = dist.entropy()

        x = self.conv_v_0(state_Bs)
        x = F.relu(x)
        x = self.conv_v_1(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.linear_v_0(x)
        x = F.relu(x)
        x = x.view(state_Ttbs.size(1), state_Ttbs.size(0) * self.batch_size, self.space)  # tBh
        x, _ = self.GRU_v(x)
        x = x.view(state_Ttbs.size(1), state_Ttbs.size(0), self.batch_size, self.space)  # tTbh
        x = x.transpose(0, 1)  # Ttbh
        x = F.relu(x)
        x = th_Tbh.unsqueeze(1).detach() * x
        state_value = self.linear_v(x).squeeze(-1)

        return action_logprobs, state_value, dist_entropy


class PPOMFOS:
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip, batch_size, inner_ep_len, tau=None):
        self.lr = lr
        #         self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCriticMFOS(state_dim, action_dim, n_latent_var, batch_size).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCriticMFOS(state_dim, action_dim, n_latent_var, batch_size).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()
        self.tau = tau
        self.inner_ep_len = inner_ep_len

    def update(self, memory):
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        global_discounted = 0
        for t, reward in enumerate(reversed(memory.rewards)):
            if t != 0 and t % self.inner_ep_len == 0:
                discounted_reward = global_discounted
            discounted_reward = reward + (self.gamma * discounted_reward)
            global_discounted = reward.mean(0, keepdim=True) + self.gamma * global_discounted
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.stack(rewards).detach()
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)  # (Tt)b

        # convert list to tensor
        old_states = torch.stack(memory.states_traj).to(device).detach()  # T,t,b,s
        del memory.states_traj[:]
        old_actions = torch.stack(memory.actions_traj).to(device).detach()
        del memory.actions_traj[:]
        old_logprobs = torch.stack(memory.logprobs_traj).to(device).detach().flatten(end_dim=-2)
        del memory.logprobs_traj[:]

        del memory.rewards[:]
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            logprobs = logprobs.flatten(end_dim=-2)

            state_values = state_values.flatten(end_dim=-2)
            dist_entropy = dist_entropy.flatten(end_dim=-2)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach()).squeeze()

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards)  # - 0.01*dist_entropy
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

    def save(self, filename):
        torch.save(
            {
                "actor_critic": self.policy.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            filename,
        )

    def load(self, filename):
        checkpoint = torch.load(filename)
        self.policy.load_state_dict(checkpoint["actor_critic"])
        self.policy_old.load_state_dict(checkpoint["actor_critic"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
