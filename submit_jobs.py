with open("template_sh", "r") as f:
    template = f.read()


def get_python_cmd_spiral(layers, seed, spiral, lr=None, epochs=2000, dun=False):
    cmd = f"python supervised_spiral.py --layers {layers} --epochs {epochs} --seed {seed} --spiral {spiral}"
    if lr is not None:
        cmd += " --lr " + str(lr)

    if dun:
        cmd += f" --dun"
    return cmd

run_shs = []

for spiral_R in [0, 1, ]:#2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]:
    for SEED in [0, 1, 2, 3, 4]:
        LS = [-1]#, 4, 3, 2, 1, "DUN10", 0, 5, 6, 7, 10, "DUN15", "DUN20"]

        N_PARTS = 1
        for part_id in range(N_PARTS):
            tmp = template
            for L in LS[part_id::N_PARTS]:
                if type(L) == int:
                    tmp += "\n" + get_python_cmd_spiral(L, SEED, spiral_R, lr=0.005, epochs=4000)
                else:
                    tmp += "\n" + get_python_cmd_spiral(L[3:], SEED, spiral_R, lr=0.005, epochs=4000,  dun=True)

            run_shs.append(tmp)

# machines = ['statler', 'waldorf']*2 + ['bobo', 'rizzo', 'yolanda', 'floyd', 'janice']
machines = {
    "yolanda": 32, # fast
    # "rizzo": 25, # rizzo is slow
    "bobo": 24, # fast
    "floyd": 24,
    "statler": 38,
    "waldorf": 38,
    "janice": 10,
}

import itertools

machines = list(itertools.chain(*[[k]*v for k,v in machines.items()]))
# machines = ['bobo', 'floyd']
# for L in [-1, 1, 2, 3, 5]:
#     tmp = template
#     tmp += "\n" + get_python_cmd_mnist(L, seed=0)
#     run_shs.append(tmp)

import random

for j, run_sh in enumerate(run_shs):
    run_sh = run_sh.replace("MACHINE", machines[j % len(machines)])
    i = random.randrange(100000)
    fn = "job_scripts/run-tmp-%d.sh" % i
    with open(fn, "w") as f:
        f.write(run_sh)

    import os

    os.system("sbatch " + fn)



    # print("Has started " + fn)
