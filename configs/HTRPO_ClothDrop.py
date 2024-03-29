from torch.nn import functional as F

HTRPOconfig = {
    'cg_damping': 1e-3,
    'reward_decay': 0.98,
    'GAE_lambda': 0.,
    'max_kl_divergence': 5e-5,
    'entropy_weight': 0,
    'per_decision': True,
    'weighted_is': True,
    'using_active_goals' : True,
    'hidden_layers': [256, 256, 256],
    'hidden_layers_v': [256, 256, 256],
    'max_grad_norm': None,
    'lr_v': 5e-4,
    'iters_v':20,
    # for comparison with HPG
    'lr': 5e-4,
    # NEED TO FOCUS ON THESE PARAMETERS
    'using_hpg': False,
    # 'steps_per_iter': 1600,
    'steps_per_iter': 3200,
    'sampled_goal_num': 100,
    'value_type': 'FC',
    'using_original_data': False,
    # 'act_func': F.tanh,
    # 'out_act_func': F.tanh,
    'act_func': F.elu,
    'out_act_func': F.elu,
    'using_kl2':True,
    'using_obgmgs_goals': True
}
HTRPOconfig['memory_size'] = HTRPOconfig['steps_per_iter']
