with open("template_sh_cifar", "r") as f:
    template = f.read()


def get_python_cmd(layers, seed, epochs=500, categories=""):
    cmd = f"python supervised-cifar10.py --layers {layers} --epochs {epochs} --seed {seed}"
    if categories:
        cmd += f" --categories {categories}"
    return cmd

run_shs = []


# for layers in [10]:
# for layers in [-1, 5, 10, 20, 30]:
# for categories in ["49", "28", "27", "02", "34"]:
#     tmp = template
#     for SEED in [0, 1, 2]:
#         # if layers == -1 and SEED == 0:
#         #     continue
#         tmp += "\n" + get_python_cmd(-1, SEED, categories=categories)
#     run_shs.append(tmp)

for SEED in [0, 1, 2]:
    tmp = template
    for layers in [-1]:
        tmp += "\n" + get_python_cmd(layers, SEED)
    run_shs.append(tmp)

import itertools
import random

for j, run_sh in enumerate(run_shs):
    i = random.randrange(100000)
    fn = "run-tmp-%d.sh" % i
    with open(fn, "w") as f:
        f.write(run_sh)

    import os

    os.system("sbatch " + fn)

    # print("Has started " + fn)
