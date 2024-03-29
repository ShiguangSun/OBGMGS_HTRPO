import torch.nn.functional as F

HTRPOconfig = {
    'reward_decay': 0.98,
    'cg_damping': 1e-3,
    'GAE_lambda': 0.,
    'max_kl_divergence': 2e-5,
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
    'steps_per_iter': 1600,
    'sampled_goal_num': 100,
    'value_type': 'FC',
    'act_func': F.tanh,
    'using_original_data': False,
    'using_kl2':True,
    'using_hgf_goals': True,
    'KL_esti_method_for_TRPO': 'origin',
    'using_obgmgs_goals': True

}
HTRPOconfig['memory_size'] = HTRPOconfig['steps_per_iter']
