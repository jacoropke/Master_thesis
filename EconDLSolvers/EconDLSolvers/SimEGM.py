import numpy as np
import torch

# for distributed training
from torch.nn.parallel import DistributedDataParallel
import torch.distributed
import torch.nn.functional as F
import matplotlib.pyplot as plt
# local
from . import auxilliary as aux
import time

scheduler_step = aux.scheduler_step

####################
# setup and create #
####################

def setup(model):
	""" setup training parameters"""

	train = model.train
	
	# a. not used
	train.Nneurons_value = None
	train.learning_rate_value = None
	train.learning_rate_value_decay = None
	train.learning_rate_value_min = None
	train.Nepochs_value = None
	train.Delta_epoch_value = None
	train.epoch_value_min = None
	train.tau = None
	train.use_target_policy = None
	train.use_target_value = None	
	train.clip_grad_value = None
	train.store_pd = True # store pd states in replay buffer
	train.use_target_policy = True




	train.N_policy_NN = None

	train.tau = 1.0


	model.info['time.compute_target'] = 0.0

def create_NN(model):
	""" create neural nets """

	aux.create_NN(model,value=False)


#########
# solve #
#########

def update_NN(model):
	""" update neural net parameters """

	# a. unpack
	train = model.train
	par = model.par

	# b. sample
	batch = model.rep_buffer.sample(train.batch_size)
	states_pd = batch.states_pd
	states = batch.states
	
	if train.terminal_actions_known: 
		states_pd = states_pd[:-1]
		states = states[:-1]
	states_pd_zeros = states_pd.clone()
	states_pd_zeros[...,0] = par.a_low[:-1][:,None] * torch.ones_like(states_pd_zeros[...,0])
	
	# c. generate target data
	t0 = time.perf_counter()
	c_target, state_pre_pd_endo, indicator = EGM_target(model,states_pd, states, states_pd_zeros)
	model.info['time.compute_target'] += time.perf_counter() - t0

	# c. update policy
	aux.train_policy(model,policy_loss_f,state_pre_pd_endo,c_target, states_pd,indicator)

	if train.use_target_policy: aux.update_target_policy_network(model)
	

def EGM_target(model,states_pd, states, states_pd_zeros):
	""" compute target """
	
	# a. unpack
	par = model.par
	train = model.train
	if train.use_target_policy:
		policy_NN = model.policy_NN_target
	else:
		policy_NN = model.policy_NN
	

	with torch.no_grad():

		m_sample = states[...,0]
		

		# a. given post-decision state compute future state
		state_plus = model.state_trans(states_pd,train.quad)
		states_plus_zeros = model.state_trans(states_pd_zeros,train.quad)

		if train.update_quad_w:
			quad_w = model.quad_w_update(state_plus)
		else:
			quad_w = train.quad_w[None,None,:]
			
		wealth_pd = states_pd[...,0]
		wealth_pd_zeros = states_pd_zeros[...,0]

		# b. compute c_plus
		actions_plus_bef = model.eval_policy(policy_NN,state_plus[:-1],t0=1)
		actions_plus_terminal = model.terminal_actions(state_plus[-1:])
		actions_plus = torch.cat((actions_plus_bef, actions_plus_terminal),dim=0)
		outcomes_plus = model.outcomes(state_plus,actions_plus,t0=1)
		c_plus = outcomes_plus[...,0] # assume c is first outcome

		actions_plus_zeros_bef = model.eval_policy(policy_NN,states_plus_zeros[:-1],t0=1)
		actions_plus_zeros_terminal = model.terminal_actions(states_plus_zeros[-1:])
		actions_plus_zeros = torch.cat((actions_plus_zeros_bef, actions_plus_zeros_terminal),dim=0)
		outcomes_plus_zeros = model.outcomes(states_plus_zeros,actions_plus_zeros,t0=1)
		c_plus_zeros = outcomes_plus_zeros[...,0] # assume c is first outcome



		# c. get future marginal utility
		mu_plus = model.marg_util(c_plus,par)
		mu_plus_zeros = model.marg_util(c_plus_zeros,par)

		marg_bequest = model.marg_bequest(wealth_pd)
		marg_bequest_zeros = model.marg_bequest(wealth_pd_zeros)

		# d. expected marginal utility
		mu_plus_exp = torch.sum(mu_plus*quad_w,dim=-1)
		mu_plus_exp_zeros = torch.sum(mu_plus_zeros*quad_w,dim=-1)

		q = par.s_p_torch[:-1,None] * par.R * mu_plus_exp + (1-par.s_p_torch[:-1,None]) * marg_bequest
		q_zeros = par.s_p_torch[:-1,None] * par.R * mu_plus_exp_zeros + (1-par.s_p_torch[:-1,None]) * marg_bequest_zeros 


		# e. compute target
		c_target = model.inv_marg_util(par.beta*q,par)
		c_target_zeros = model.inv_marg_util(par.beta*q_zeros,par)

		# f. compute state_pre_pd_endo
		m_endo = wealth_pd + c_target
		m_endo_zeros = wealth_pd_zeros  +  c_target_zeros
		state_pre_pd_endo = torch.cat((m_endo[...,None],states_pd[...,1:]),dim=-1)


		c_target_constrained = (m_sample - par.a_low[:-1][:,None])[...,None]
		state_pre_pd_endo_constrained = torch.cat((m_sample[...,None],states_pd[...,1:]),dim=-1)

		c_target_all = torch.cat((c_target[...,None], c_target_constrained), dim=1)
		state_pre_pd_endo_all = torch.cat((state_pre_pd_endo,state_pre_pd_endo_constrained ),dim=1)

		# Create indicator for unconstrained agents
		indicator_unconstrained = torch.ones_like(m_sample, dtype=torch.bool)

		# Create indicator for constrained agents
		indicator_constrained = m_sample < m_endo_zeros

		# Concatenate the two indicators along dim=1 to match the shape of c_target_all and state_pre_pd_endo_all
		indicator_all = torch.cat((indicator_unconstrained, indicator_constrained), dim=1)



	return c_target_all, state_pre_pd_endo_all, indicator_all




def policy_loss_f(model, states, target, states_pd,indicator):
	"""Policy loss with monotonicity regularization"""

	# a. unpack
	train = model.train

	# b. actions and consumption today
	actions = model.eval_policy(model.policy_NN, states)
	outcomes = model.outcomes(states, actions)
	c = outcomes[..., 0][...,None]  # Predicted consumption

	# c. main MSE loss
	policy_loss = F.mse_loss(c[indicator], target[indicator])
	total_loss = policy_loss

	return total_loss




