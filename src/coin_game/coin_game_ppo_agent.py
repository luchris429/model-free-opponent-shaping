import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]


class ActorCritic(nn.Module):
    def __init__(self, input_shape, action_dim, n_latent_var, n_out_channels):
        super(ActorCritic, self).__init__()

        self.conv_a_0 = nn.Conv2d(input_shape[0], n_out_channels, kernel_size=3, stride=1, padding="same", padding_mode="circular")
        self.conv_a_1 = nn.Conv2d(n_out_channels, n_out_channels, kernel_size=3, stride=1, padding="same", padding_mode="circular")

        self.linear_a_0 = nn.Linear(n_out_channels * input_shape[1] * input_shape[2], n_latent_var)
        self.linear_a_1 = nn.Linear(n_latent_var, action_dim)

        self.conv_v_0 = nn.Conv2d(input_shape[0], n_out_channels, kernel_size=3, stride=1, padding="same", padding_mode="circular")
        self.conv_v_1 = nn.Conv2d(n_out_channels, n_out_channels, kernel_size=3, stride=1, padding="same", padding_mode="circular")

        self.linear_v_0 = nn.Linear(n_out_channels * input_shape[1] * input_shape[2], n_latent_var)
        self.linear_v_1 = nn.Linear(n_latent_var, 1)

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
        return F.softmax(self.linear_a_1(x).squeeze(0), dim=-1)

    def forward_v(self, state_bs):
        x = self.conv_v_0(state_bs)
        x = F.relu(x)
        x = self.conv_v_1(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.linear_v_0(x)
        x = F.relu(x)
        return self.linear_v_1(x).squeeze(0)

    def act(self, state_bs, memory):
        action_probs_ba = self.forward_a(state_bs)
        value_pred_b = self.forward_v(state_bs)

        dist = Categorical(action_probs_ba)
        action_oba = dist.sample()

        memory.states.append(state_bs)
        memory.actions.append(action_oba)
        memory.logprobs.append(dist.log_prob(action_oba))

        return action_oba.squeeze(0)

    def evaluate(self, state_bs, action_oba):
        action_probs_ba = self.forward_a(state_bs)
        dist = Categorical(action_probs_ba)

        action_logprobs = dist.log_prob(action_oba)
        dist_entropy = dist.entropy()

        state_value = self.forward_v(state_bs)

        return action_logprobs, torch.squeeze(state_value), dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip, tau=None):
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(state_dim, action_dim, n_latent_var, n_latent_var // 4).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(state_dim, action_dim, n_latent_var, n_latent_var // 4).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()
        self.tau = tau

    def update(self, memory):
        # Monte Carlo estimate of rewards:
        rewards = []
        discounted_reward = 0
        for reward in reversed(memory.rewards):
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.stack(rewards).squeeze(-1)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        rewards = rewards.flatten(end_dim=1)

        # convert list to tensor
        old_states = torch.stack(memory.states).detach().flatten(end_dim=1)
        old_actions = torch.stack(memory.actions).detach().flatten(end_dim=1)
        old_logprobs = torch.stack(memory.logprobs).detach().flatten(end_dim=1)

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

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
