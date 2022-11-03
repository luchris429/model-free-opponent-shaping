import torch
import os
import json
from coin_game_envs import CoinGameGPU
from coin_game.coin_game_ppo_agent import PPO, Memory
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--exp-name", type=str, default="")
args = parser.parse_args()

if __name__ == "__main__":
    ############## Hyperparameters ##############
    batch_size = 512  # 8192 #, 32768
    state_dim = [4, 3, 3]
    action_dim = 4
    n_latent_var = 8  # number of variables in hidden layer

    lr = 0.005
    betas = (0.9, 0.999)
    gamma = 0.96  # discount factor

    traj_length = 32

    K_epochs = 80  # update policy for K epochs
    eps_clip = 0.2  # clip parameter for PPO
    use_gae = False

    inner_ep_len = 32
    num_steps = 512  # , 500
    max_episodes = num_steps // inner_ep_len  # max training episodes
    name = args.exp_name

    print(f"RUNNING NAME: {name}")
    if not os.path.isdir(name):
        os.mkdir(name)
        with open(os.path.join(name, "commandline_args.txt"), "w") as f:
            json.dump(args.__dict__, f, indent=2)
    #############################################

    memory_0 = Memory()
    ppo_0 = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)
    memory_1 = Memory()
    ppo_1 = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)

    print(lr, betas)
    print(sum(p.numel() for p in ppo_0.policy_old.parameters() if p.requires_grad))
    # logging variables
    running_reward = 0
    avg_length = 0
    timestep = 0
    rew_means = []

    # env
    env = CoinGameGPU(batch_size=batch_size, max_steps=inner_ep_len - 1)

    # training loop
    for i_episode in range(1, max_episodes + 1):
        state = env.reset()
        running_reward = 0
        opp_running_reward = 0
        p1_num_opp, p2_num_opp, p1_num_self, p2_num_self = 0, 0, 0, 0
        for t in range(inner_ep_len):
            # Running policy_old:
            with torch.no_grad():
                action_0 = ppo_0.policy_old.act(state[0].detach(), memory_0)
                action_1 = ppo_1.policy_old.act(state[1].detach(), memory_1)

            state, reward, done, info_2 = env.step([action_0.detach(), action_1.detach()])

            running_reward += reward[0].detach()
            opp_running_reward += reward[1].detach()
            memory_0.rewards.append(reward[0].detach())
            memory_1.rewards.append(reward[1].detach())
            if info_2 is not None:
                p1_num_opp += info_2[2]
                p2_num_opp += info_2[1]
                p1_num_self += info_2[3]
                p2_num_self += info_2[0]

        ppo_0.update(memory_0)
        ppo_1.update(memory_1)
        memory_0.clear_memory()
        memory_1.clear_memory()

        rew_means.append(
            {
                "episode": i_episode,
                "rew": running_reward.mean().item(),
                "opp_rew": opp_running_reward.mean().item(),
                "p1_opp": p1_num_opp.float().mean().item(),
                "p2_opp": p2_num_opp.float().mean().item(),
                "p1_self": p1_num_self.float().mean().item(),
                "p2_self": p2_num_self.float().mean().item(),
            }
        )
        print(rew_means[-1])

    ppo_0.save(os.path.join(name, f"{i_episode}_0.pth"))
    ppo_1.save(os.path.join(name, f"{i_episode}_1.pth"))
    with open(os.path.join(name, f"out_{i_episode}.json"), "w") as f:
        json.dump(rew_means, f)
    print(f"SAVING! {i_episode}")
