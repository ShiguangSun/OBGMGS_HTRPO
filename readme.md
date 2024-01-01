# Optimal Bipartite Graph Matching-Based Goal Selection for Policy-Based Hindsight Learning

## Introduction
This is a deep reinforcement learning package including our newly proposed algorithm OBGMGS-HTRPO. See the original HTRPO code: (https://github.com/HTRPOCODES/HTRPO-v2)

## Requirements
The same requirements as HTRPO.

If you want to run deformable object manipulation tasks, you should first install [SoftGym](https://github.com/Xingyu-Lin/softgym).

## Examples
For running continuous envs (e.g. FetchPush-v1) with HTRPO algorithm:
```bash
python main.py --alg HTRPO --env FetchPush-v1 --num_steps 2000000 --num_evals 100 --eval_interval 19200 --notearlyend(--cpu)
```

For running deformable object manipulation envs:
```bash
CUDA_VISIBLE_DEVICES=1 python main.py --num_envs 1 --alg HTRPO --env ClothFold-v0 --seed 1 --eval_interval 19200 --num_steps 1000000 --num_eval 100 --notearlyend
```

--cpu is used only when you want to train the policy using CPU, which will be much slower than using GPU. And deformable object manipulation tasks have to use gpu.

**Note**: 
We propose OBGMGS for policy-based hindsight learning. To run OBGMGS-HTRPO, you need to follow the above instruction. To run OBGMGS-HPG, you only need to modify the hyperparameter "using_hpg" to "True" in the corresponding config file (e.g. for FetchPush-v1, the config file is configs/HTRPO_FetchPushv1.py). To run original HTRPO and HPF, you need to modify the hyperparameter "using_obgmgs_goals" to "False" in the corresponding config file.
