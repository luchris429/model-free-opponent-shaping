import math
import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BatchedPolicies(nn.Module):
    def __init__(self, state_dim, action_dim, num_species, std=2, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.l_1 = nn.Parameter(torch.empty((state_dim, 256), **factory_kwargs))
        self.b_1 = nn.Parameter(torch.empty((256), **factory_kwargs))

        self.l_2 = nn.Parameter(torch.empty((256, action_dim), **factory_kwargs))
        self.b_2 = nn.Parameter(torch.empty((action_dim), **factory_kwargs))

        nn.init.kaiming_uniform_(self.l_1, a=math.sqrt(5))  # weight init
        nn.init.kaiming_uniform_(self.l_2, a=math.sqrt(5))  # weight init

        fan_in_1, _ = nn.init._calculate_fan_in_and_fan_out(self.l_1)
        bound = 1 / math.sqrt(fan_in_1)
        nn.init.uniform_(self.b_1, -bound, bound)  # bias init

        fan_in_2, _ = nn.init._calculate_fan_in_and_fan_out(self.l_2)
        bound = 1 / math.sqrt(fan_in_2)
        nn.init.uniform_(self.b_2, -bound, bound)  # bias init

        self.num_species = num_species
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.std = std

    def create_species(self):
        self.l1_noise = torch.normal(mean=0, std=self.std, size=(self.num_species, self.state_dim, 256), device=device)
        self.b1_noise = torch.normal(mean=0, std=self.std, size=(self.num_species, 256), device=device)
        self.l2_noise = torch.normal(mean=0, std=self.std, size=(self.num_species, 256, self.action_dim), device=device)
        self.b2_noise = torch.normal(mean=0, std=self.std, size=(self.num_species, self.action_dim), device=device)

        self.l1_noise[-1] = 0.0
        self.b1_noise[-1] = 0.0
        self.l2_noise[-1] = 0.0
        self.b2_noise[-1] = 0.0

        self.fl1_nsd = self.l_1 + self.l1_noise
        self.fb1_nd = self.b_1 + self.b1_noise
        self.fl2_nda = self.l_2 + self.l2_noise
        self.fb2_na = self.b_2 + self.b2_noise

    def forward(self, state_nbs):
        x = torch.einsum("nsd,nbs->nbd", self.fl1_nsd, state_nbs)
        x += self.fb1_nd.unsqueeze(1)  # ?
        x = torch.tanh(x)

        x = torch.einsum("nda,nbd->nba", self.fl2_nda, x)
        x += self.fb2_na.unsqueeze(1)

        return x

    def forward_eval(self, state_bs, idx=None):
        if not idx:
            x = torch.einsum("sd,bs->bd", self.l_1, state_bs)
            x += self.b_1.unsqueeze(0)
            x = torch.tanh(x)
            x = torch.einsum("da,bd->ba", self.l_2, x)
            x += self.b_2.unsqueeze(0)
        else:
            x = torch.einsum("sd,bs->bd", self.fl1_nsd[idx], state_bs)
            x += self.fb1_nd[idx].unsqueeze(0)
            x = torch.tanh(x)
            x = torch.einsum("da,bd->ba", self.fl2_nda[idx], x)
            x += self.fb2_na[idx].unsqueeze(0)

        return x
