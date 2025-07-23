import torch
import torch.nn.functional as F

# local
from . import auxilliary as aux

scheduler_step = aux.scheduler_step
import matplotlib.pyplot as plt
from copy import deepcopy
import time
#########
# setup #
#########

def setup(model):
	""" setup training parameters"""

	train = model.train

	# a. targets
	train.tau = 0.2 # target smoothing coefficient
	train.use_target_policy = False
	train.use_target_value = True

	# b. misc
	train.store_pd = True # store pd states in replay buffer
	train.store_pd_conditional = True

	# c. FOC
	train.start_train_policy = 50
	train.use_FOC = False

	train.Ngrid = 5
	train.m_pd_min_factor = 0.5
	train.m_pd_max_factor = 2.0
	train.m_max_factor = 1.15
	train.m_min_factor = 1/train.m_max_factor

	train.Nforgrid = 50

	train.learning_rate_value = 1e-4
	# train.m_max_factor = 1.1


def create_NN(model):
	""" create neural nets """

	aux.create_NN(model,Noutputs_value=1)

###########
# solving #
###########

def update_NN(model):
	""" update neural networks """

	# unpack
	train = model.train
	par = model.par


	# a. sample
	batch = model.rep_buffer.sample(train.batch_size)
	states, states_pd,states_pd_conditional = batch.states, batch.states_pd, batch.states_pd_conditional

	# b. remove last period observation
	states_pd = states_pd[:-1]
	states_pd_conditional = states_pd_conditional[:-1]
	if train.terminal_actions_known: states = states[:-1]

	# c. create conditional grid
	states_pd_conditional_grid = create_state_grid_data_simulated(model,states_pd_conditional[:,:train.Nforgrid],conditional=True,pd=True)

	# b. update value
	if torch.cuda.is_available(): torch.cuda.synchronize(device=train.device)
	aux.train_value(model,compute_target,value_loss_f,states_pd)
	if torch.cuda.is_available(): torch.cuda.synchronize(device=train.device)

	# c. update target value
	if train.use_target_value: aux.update_target_value_network(model)

	
	if train.k >= train.start_train_policy:

		# d. compute consumption target and endogenous grid fom EGM-step
		if torch.cuda.is_available(): torch.cuda.synchronize(device=train.device)
		t0 = time.perf_counter()
		c_target, m_endo_raw = EGM_target_nocommon(model,states_pd_conditional_grid)
		
		# e. reshape and grab middle point of endogenous grid
		m_endo_raw = m_endo_raw.reshape(par.T-1, train.Nforgrid, train.Ngrid,par.NDC)
		middle_point = train.Ngrid//2
		m_endo_raw = m_endo_raw[:,:,middle_point,:]
		if torch.cuda.is_available(): torch.cuda.synchronize(device=train.device)
		model.info['time.EGM_target'] += time.perf_counter() - t0


		# f. combine endogeus grid with remaining states
		states_DC = states[:,:train.Nforgrid,None,1:].expand(-1,-1,par.NDC,-1)
		states_endo = torch.cat((m_endo_raw.unsqueeze(3),states_DC),dim=3)

		check = torch.sum(c_target==0)
		if check > 0:
			raise ValueError('consumption target was equal to zero')

		# g. update policy
		aux.train_policy(model,policy_loss_f_nocommon,states_endo, c_target)
		if torch.cuda.is_available(): torch.cuda.synchronize(device=train.device)


		# h. update target policy
		if train.use_target_policy:
			aux.update_target_network(train.tau,model.policy_NN,model.policy_NN_target)


def create_state_grid_data_simulated(model,states, conditional=False,pd=False):
	""" create state grid data """

	# a. unpack
	par = model.par
	train = model.train
	m = states[...,0,:]
	states_nom = states[...,1:,:]
	N_grid = train.Ngrid  # Number of grid points
	T, N, N_states,_ = states.shape
	
	# b. m_min and max point in grid
	m_min = (m - train.m_minus).clamp(0.0)
	m_max = m + train.m_plus
	expand_tuple = (T, N, N_grid, par.NDC)
	m_min_expanded = m_min[...,None,:].expand(expand_tuple)  # Shape: (T, N, N_grid)
	m_max_expanded = m_max[...,None,:].expand(expand_tuple)  # Shape: (T, N, N_grid)

	# c. Create a normalized grid ranging from 0 to 1
	linspace_normalized = torch.linspace(0, 1, N_grid, device=states.device, dtype=states.dtype)  # Shape: (N_grid,)
	m_grids = m_min_expanded + (m_max_expanded - m_min_expanded) * linspace_normalized[None,None,:,None]  # Shape: (T, N, N_grid)
	reshape_tuple = (T, N * N_grid, par.NDC)
	m_grids = m_grids.reshape(reshape_tuple)  # Shape: (T, N * N_grid, 1)

	# d. combine with remaining states
	states_nom = torch.repeat_interleave(states_nom, N_grid, dim=1)
	concat_dim = -2
	states_grid = torch.cat((m_grids[:,:,None,:], states_nom), dim=concat_dim)

	return states_grid


###################
# value training #
###################

def choice_prob(par,v0):
	""" compute choice probabilities """

	choice_probs = torch.zeros_like(v0)
	vmax = torch.max(v0,dim=-1,keepdim=True)[0]
	v_vmax = (v0-vmax)/par.sigma_eps
	for d in range(par.NDC):
		choice_probs[...,d] = torch.exp(v_vmax[...,d])/torch.sum(torch.exp(v_vmax),dim=-1)
	
	return choice_probs

def compute_target(model,states_pd):
	""" compute target """

	# a. unpack
	par = model.par
	train = model.train
	dtype = train.dtype
	device = train.device
	
	if train.use_target_value:
		value_NN = model.value_NN_target
	else:
		value_NN = model.value_NN

	if train.use_target_policy:
		policy_NN = model.policy_NN_target
	else:
		policy_NN = model.policy_NN

	# b. compute target
	with torch.no_grad():

		# i. future states
		states_plus = model.state_trans(states_pd,train.quad) # shape = (T,N,Nquad,Nstates)

		# ii. future actions
		actions_plus_before = model.eval_policy(policy_NN,states_plus[:-1],t0=1)
		actions_plus_after = model.terminal_actions(states_plus[-1:])
		actions_plus = torch.cat([actions_plus_before,actions_plus_after],dim=0)

		# iii. future reward
		outcomes_plus = model.outcomes(states_plus,actions_plus,t0=1) # shape = (T,N,Nquad,NDC)
		reward_plus = model.reward(states_plus,actions_plus,outcomes_plus,t0=1) # shape = (T,N,Nquad,NDC)

		# iv. future post-decision states
		states_pd_plus = model.state_trans_pd(states_plus,actions_plus,outcomes_plus,t0=1).permute(0,1,2,4,3) # swap last two dimensions

		# v. future post-decision value
		value_pd_plus_before = model.eval_value_pd(value_NN,states_pd_plus[:-1],t0=1)[...,0]
		value_pd_plus_after = model.terminal_reward_pd(states_pd_plus[-1:])[...,0]
		value_pd_plus = torch.cat([value_pd_plus_before,value_pd_plus_after],dim=0)

		# vi. future value function
		discount_factor = model.discount_factor(states_plus,t0=1)[...,None]
		value_plus = (reward_plus + discount_factor*value_pd_plus).reshape(-1,train.Nquad,par.NDC)


		# vii. expected future value
		target_value_quad = par.sigma_eps*torch.logsumexp(value_plus/par.sigma_eps,dim=-1,keepdim=False)
		target_value = torch.sum(target_value_quad*train.quad_w[None,:],dim=-1,keepdim=True) # target post-decision value function

	return target_value

def value_loss_f(model,target,states_pd):
	""" value loss """

	# a. unpack
	train = model.train
	value_NN = model.value_NN

	# b. prediction
	# print(states_pd.shape)
	pred = model.eval_value_pd(value_NN,states_pd)
	value_pd_pred = pred[...,0].reshape(-1,1)

	# c. targets
	target_val = target[...,0].reshape(-1,1)

	# d. loss
	loss = F.mse_loss(value_pd_pred,target_val)

	return loss

###################
# policy training #
###################

def choice_prob(par,v0):
	""" compute choice probabilities """

	choice_probs = torch.zeros_like(v0)
	vmax = torch.max(v0,dim=-1,keepdim=True)[0]
	v_vmax = (v0-vmax)/par.sigma_eps
	for d in range(par.NDC):
		choice_probs[...,d] = torch.exp(v_vmax[...,d])/torch.sum(torch.exp(v_vmax),dim=-1)
	
	return choice_probs

def EGM_target_nocommon(model,states_pd_grid):
	""" compute target """
	
	# a. unpack
	par = model.par
	train = model.train
	policy_NN = model.policy_NN
	if train.N_value_NN is None:
		value_NN = model.value_NN
	else:
		value_NN = model.value_NNs
	

	with torch.no_grad():
		
		# b. given post-decision state compute future state
		state_plus = model.state_trans(states_pd_grid.permute(0,1,3,2),train.quad)

		# c. compute consumption at t+1
		actions_plus_before = model.eval_policy(policy_NN,state_plus[:-1],t0=1)
		actions_plus_after = model.terminal_actions(state_plus[-1:])
		actions_plus = torch.cat([actions_plus_before,actions_plus_after],dim=0)
		outcomes_plus = model.outcomes(state_plus,actions_plus,t0=1)
		c_plus = outcomes_plus # assume c is first outcome

		# d. get future marginal utility
		mu_plus = model.marg_util_c(c_plus,state_plus) # not general 

		# e. compute next-period choice probabilities
		# i. compuce choice-specific value functions using pd neural network
		states_pd_plus = model.state_trans_pd(state_plus,actions_plus,outcomes_plus,t0=1).permute(0,1,2,3,5,4)
		value_pd_plus_before = model.eval_value_pd(value_NN,states_pd_plus[:-1],t0=1)[...,0]
		value_pd_plus_after = model.terminal_reward_pd(states_pd_plus[-1:])[...,0]
		value_pd_plus = torch.cat([value_pd_plus_before,value_pd_plus_after],dim=0)
		discount_factor = par.beta
		reward_plus = model.reward(state_plus,actions_plus,outcomes_plus,t0=1)
		value_plus = (reward_plus + discount_factor*value_pd_plus)
		# ii. compute CCPs
		choice_probs = choice_prob(par,value_plus)

		# f. expected marginal utility
		mu_plus_exp = torch.sum(mu_plus*choice_probs,dim=-1) # taste shocks
		mu_plus_exp = torch.sum(mu_plus_exp*train.quad_w[None,None,:,None],dim=2) # other shocks

		# g. use inverted marginal utility to compute consumption
		c_target_raw = (par.beta*par.R*mu_plus_exp)**(-1/par.rho)

		# h. compute endogenous cash-on-hand grid
		m_endo_raw = states_pd_grid[...,0,:] + c_target_raw

		# j. compute value_pd grid
		v_pd_raw_before = model.eval_value_pd(value_NN,states_pd_grid[:-1].permute(0,1,3,2))[...,0]
		v_pd_raw_after = model.terminal_reward_pd(states_pd_grid[-1:].permute(0,1,3,2))[...,0]
		v_pd_raw = torch.cat([v_pd_raw_before,v_pd_raw_after],dim=0)

		# k. upper envelope
		t0 = time.perf_counter()
		if torch.cuda.is_available(): torch.cuda.synchronize(device=train.device)
		c_target = upper_envelope_nocommon_sim(model,states_pd_grid.permute(0,1,3,2), c_target_raw,m_endo_raw,v_pd_raw)
		if torch.cuda.is_available(): torch.cuda.synchronize(device=train.device)
		model.info['time.upper_envelope'] += time.perf_counter() - t0


	return c_target, m_endo_raw




def policy_loss_f_nocommon(model,states,target):
	""" policy loss """

	# a. unpack
	train = model.train

	# b. compute actions and pick out right actions given states structure
	actions_ = model.eval_policy(model.policy_NN,states)
	actions_work = actions_[...,0,0]
	actions_retired = actions_[...,1,1]
	actions = torch.cat((actions_work[...,None],actions_retired[...,None]),dim=-1)
	
	# c. compute policy loss
	outcomes = actions
	policy_loss = F.mse_loss(outcomes,target)

	return policy_loss

def upper_envelope_nocommon_sim(model, states_pd_grid, c_target_raw, m_endo_raw, v_pd_raw):
	""" Upper Envelope (Fully Vectorized) """

	# a. Unpack parameters
	par = model.par
	train = model.train

	# a. reshape to make grid points explicit
	states_pd_grid = states_pd_grid.reshape(par.T-1, train.Nforgrid,train.Ngrid, par.NDC, par.Nstates_pd)
	c_target_raw = c_target_raw.reshape(par.T-1, train.Nforgrid, train.Ngrid, par.NDC)
	v_pd_raw = v_pd_raw.reshape(par.T-1, train.Nforgrid, train.Ngrid, par.NDC)
	m_endo_raw = m_endo_raw.reshape(par.T-1, train.Nforgrid, train.Ngrid, par.NDC)
	
	# b. prepare array of zeros
	c_target_cleaned = torch.zeros((par.T-1, train.Nforgrid, par.NDC),device=train.device,dtype=train.dtype)

	# c. get grid points for interpolation purposes
	m_low = m_endo_raw[:, :, :-1]
	m_high = m_endo_raw[:, :, 1:]
	m_pd_low = states_pd_grid[:, :, :-1, :,0]
	m_pd_high = states_pd_grid[:, :, 1:, :,0]
	c_low = c_target_raw[:, :, :-1]
	c_high = c_target_raw[:, :, 1:]
	v_pd_low = v_pd_raw[:, :, :-1]
	v_pd_high = v_pd_raw[:, :, 1:]

	# d. compute slopes for interpolation
	v_pd_slope = (v_pd_high - v_pd_low) / (m_pd_high - m_pd_low)
	c_slope = (c_high - c_low) / (m_high - m_low)

	# e. get m corresponding to middle grid point
	middle_grid = train.Ngrid//2
	m  = m_endo_raw[:, :, None, middle_grid,:].expand(-1,-1,train.Ngrid-1,-1)

	# f. booleans for checking whether we should interpolate
	interp_bool = (m >= m_low) & (m <= m_high)
	extrap_mask = (m > m_high[:, :, -1:])
	bool_mask = interp_bool | extrap_mask

	# g. interpolate consumption and post-decision value
	c_guess = c_low + c_slope * (m - m_low)
	m_pd_guess = m - c_guess
	v_pd_guess = v_pd_low + v_pd_slope * (m_pd_guess - m_pd_low)

	# h. compute value of different guesses
	u = model.util(c_guess)
	v_guess = u + par.beta * v_pd_guess
	v_guess = torch.where(bool_mask, v_guess, torch.full_like(v_guess, -1e10))

	# i. argamx to find best guess
	best_idx = torch.argmax(v_guess, dim=2, keepdim=True)
	c_target_cleaned_loop = torch.gather(c_guess, 2, best_idx).squeeze(2)
	
	# j. use booleans to only update where applicable
	bool_mask = torch.gather(bool_mask, 2, best_idx).squeeze(2)
	c_target_cleaned = torch.where(bool_mask, c_target_cleaned_loop, c_target_cleaned)

	return c_target_cleaned