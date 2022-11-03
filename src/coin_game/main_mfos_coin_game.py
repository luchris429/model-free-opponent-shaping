import torch
import os
import json
from coin_game_envs import CoinGamePPO
from coin_game.coin_game_mfos_agent import MemoryMFOS, PPOMFOS
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--exp-name", type=str, default="")
args = parser.parse_args()

if __name__ == "__main__":
    ############## Hyperparameters ##############
    batch_size = 512  # 8192 #, 32768
    state_dim = [7, 3, 3]
    action_dim = 4
    n_latent_var = 16  # number of variables in hidden layer
    max_episodes = 1000  # max training episodes
    log_interval = 50

    lr = 0.0002
    betas = (0.9, 0.999)
    gamma = 0.995  # discount factor
    tau = 0.3  # GAE

    traj_length = 16

    K_epochs = 16  # update policy for K epochs
    eps_clip = 0.2  # clip parameter for PPO
    use_gae = False

    inner_ep_len = 16
    num_steps = 256  # , 500

    do_sum = False

    save_freq = 50

    name = args.exp_name
    print(f"RUNNING NAME: {name}")
    if not os.path.isdir(name):
        os.mkdir(name)
        with open(os.path.join(name, "commandline_args.txt"), "w") as f:
            json.dump(args.__dict__, f, indent=2)
    #############################################

    memory = MemoryMFOS()
    ppo = PPOMFOS(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip, batch_size, inner_ep_len)
    print(lr, betas)
    print(sum(p.numel() for p in ppo.policy_old.parameters() if p.requires_grad))
    # logging variables
    running_reward = 0
    avg_length = 0
    timestep = 0
    rew_means = []

    # env
    env = CoinGamePPO(batch_size, inner_ep_len)

    # training loop
    for i_episode in range(1, max_episodes + 1):
        memory.clear_memory()
        state = env.reset()
        running_reward = 0
        opp_running_reward = 0
        p1_num_opp, p2_num_opp, p1_num_self, p2_num_self = 0, 0, 0, 0
        for t in range(num_steps):
            # Running policy_old:
            if t % inner_ep_len == 0:
                ppo.policy_old.reset(memory, t == 0)
            with torch.no_grad():
                action = ppo.policy_old.act(state.detach())
            state, reward, done, info, info_2 = env.step(action.detach())
            running_reward += reward.detach()
            opp_running_reward += info.detach()
            memory.rewards.append(reward.detach())
            if info_2 is not None:
                p1_num_opp += info_2[2]
                p2_num_opp += info_2[1]
                p1_num_self += info_2[3]
                p2_num_self += info_2[0]

        ppo.policy_old.reset(memory)
        ppo.update(memory)

        print("=" * 10)

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

        if i_episode % save_freq == 0:
            ppo.save(os.path.join(name, f"{i_episode}.pth"))
            with open(os.path.join(name, f"out_{i_episode}.json"), "w") as f:
                json.dump(rew_means, f)
            print(f"SAVING! {i_episode}")
