import os
import json
import torch
import argparse
from environments import NonMfosMetaGames

parser = argparse.ArgumentParser()
parser.add_argument("--exp-name", type=str, default="")
args = parser.parse_args()

if __name__ == "__main__":
    batch_size = 8192
    num_steps = 100
    name = args.exp_name

    print(f"RUNNING NAME: {name}")
    if not os.path.isdir(name):
        os.mkdir(name)
        with open(os.path.join(name, "commandline_args.txt"), "w") as f:
            json.dump(args.__dict__, f, indent=2)

    results = []
    for game in ["IPD", "IMP", "chicken"]:
        for p1 in ["STATIC", "NL", "LOLA", "MAMAML"]:
            for p2 in ["STATIC", "NL", "LOLA", "MAMAML"]:
                if p1 == "MAMAML" or p2 == "MAMAML":
                    for id in range(10):
                        env = NonMfosMetaGames(batch_size, game=game, p1=p1, p2=p2, mmapg_id=id)
                        env.reset()
                        running_rew_0 = torch.zeros(batch_size).cuda()
                        running_rew_1 = torch.zeros(batch_size).cuda()
                        for i in range(num_steps):
                            _, r0, r1, M = env.step()
                            running_rew_0 += r0.squeeze(-1)
                            running_rew_1 += r1.squeeze(-1)
                        mean_rew_0 = (running_rew_0.mean() / num_steps).item()
                        mean_rew_1 = (running_rew_1.mean() / num_steps).item()

                        results.append(
                            {
                                "game": game,
                                "p1": p1,
                                "p2": p2,
                                "rew_0": mean_rew_0,
                                "rew_1": mean_rew_1,
                                "mmapg_id": id,
                            }
                        )
                        print("=" * 100)
                        print(f"Done with game: {game}, p1: {p1}, p2: {p2}, id: {id}")
                        print(f"r0: {mean_rew_0}, r1: {mean_rew_1}")
                else:
                    env = NonMfosMetaGames(batch_size, game=game, p1=p1, p2=p2)
                    env.reset()
                    running_rew_0 = torch.zeros(batch_size).cuda()
                    running_rew_1 = torch.zeros(batch_size).cuda()
                    for i in range(num_steps):
                        _, r0, r1, M = env.step()
                        running_rew_0 += r0.squeeze(-1)
                        running_rew_1 += r1.squeeze(-1)
                    mean_rew_0 = (running_rew_0.mean() / num_steps).item()
                    mean_rew_1 = (running_rew_1.mean() / num_steps).item()

                    results.append({"game": game, "p1": p1, "p2": p2, "rew_0": mean_rew_0, "rew_1": mean_rew_1})
                    print("=" * 100)
                    print(f"Done with game: {game}, p1: {p1}, p2: {p2}")
                    print(f"r0: {mean_rew_0}, r1: {mean_rew_1}")
    with open(os.path.join(name, f"out.json"), "w") as f:
        json.dump(results, f)
