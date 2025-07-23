import numpy as np
import torch

from EconDLSolvers import expand_to_quad, expand_to_states, exploration, discount_factor

#####################
# policy prediction #
#####################

def predict_consumption(par,train, states,actions,t=None, t0=0):
	""" predict consumption """

	if par.policy_predict == 'savings_rate':
		wealth = states[...,0]
		if t is not None:
			m_pd = par.a_low[t] + actions[...,0] * (wealth - par.a_low[t])
		else:
			T_dim = states.shape[0]
			a_low = par.a_low[t0:T_dim+t0]
			if len(wealth.shape) == 2:
				a_low_expand = a_low[:,None].expand(wealth.shape[0],wealth.shape[1])
				m_pd = a_low_expand + actions[...,0] * (wealth - a_low_expand)
			elif len(wealth.shape) == 3:
				a_low_expand = a_low[:,None,None].expand(wealth.shape[0],wealth.shape[1],wealth.shape[2])
				m_pd = a_low_expand + actions[...,0] * (wealth - a_low_expand)
			else:
				raise ValueError('actions must be 2D or 3D')
			
		# print(m_pd.shape)
		consumption = wealth - m_pd

	elif par.policy_predict == 'consumption':
		consumption_ = actions[...,0]
		wealth = states[...,0]
		consumption = torch.clamp(consumption_, train.min_actions, wealth)

			
	return consumption

###########
# utility #
###########

def util(c,par):
	""" utility """

	return c**(1-par.gamma)/(1-par.gamma)

def marg_util_c(model,c,par):
	""" marginal utility of consumption """

	return c**(-par.gamma)

def inverse_marg_util(model,u,par):
	"""Inverse function of marginal utility of consumption """

	return u**(-1/par.gamma)



def marg_bequest(model,m_pd):
	""" marginal bequest """

	# Case I: states_pd.shape = (1,...,Nstates_pd)
	# Case II: states_pd.shape = (N,Nstates_pd)

	par = model.par
	train = model.train
	dtype = train.dtype
	device = train.device

	if par.bequest_1 != 0.0:
		bequest_neg = par.bequest_1 * (m_pd + par.bequest_2)**(-par.gamma)
		bequest_pos = par.bequest_1 * (1-par.tau_bequest) * (m_pd * (1-par.tau_bequest) + par.bequest_2)**(-par.gamma)
		bequest_util = torch.where(m_pd > 0, bequest_pos, bequest_neg)
	else:
		bequest_util = torch.zeros_like(m_pd)

	return bequest_util

###########
# reward #
###########

def outcomes(model,states,actions,t0=0,t=None):
	""" outcomes """

	par = model.par
	train = model.train

	c = predict_consumption(par,train,states,actions,t=t, t0=t0)

	m_pd = states[...,0] - c

	return torch.stack((c,m_pd),dim=-1)

def reward(model,states,actions,outcomes,t0=0,t=None):
	""" reward """

	par = model.par
	train = model.train

	# a. consumption
	c = outcomes[...,0]

	# b. utility
	u = util(c,par)

	# c. finalize
	return u 

def marginal_reward(model,states,actions,outcomes,t0=0,t=None):
	""" marginal reward """

	# a. consumption
	c = outcomes[...,0]

	# b. finalize
	return marg_util_c(c,model.par)

############
# terminal #
############

def terminal_actions(model,states):
	""" terminal actions """
	
	par = model.par
	train = model.train
	dtype = train.dtype
	device = train.device

	# wealth = torch.exp(states[...,0])
	wealth = states[...,0]
	if par.policy_predict == 'savings_rate':
		if par.bequest_1 == 0.0:
			actions = torch.zeros_like(states[...,0]).unsqueeze(-1)
		else:
			terminal_con = 1 / ((par.bequest_1*par.beta*(1-par.tau_bequest))**(1/par.gamma) + (1-par.tau_bequest)) * (wealth * (1-par.tau_bequest) + par.bequest_2)
			actions = 1-terminal_con/wealth
			actions = actions.clamp(0.0,1.0).unsqueeze(-1)

	elif par.policy_predict == 'consumption':
		if par.bequest_1 == 0.0:
			actions = wealth.unsqueeze(-1)
		else:
			actions = 1 / ((par.bequest_1*par.beta*(1-par.tau_bequest))**(1/par.gamma) + (1-par.tau_bequest)) * (wealth * (1-par.tau_bequest) + par.bequest_2)
			actions = actions.unsqueeze(-1)
	else:
		raise ValueError('policy_predict must be either savings_rate or consumption')

	return actions 


def terminal_reward_pd(model,states_pd):
	""" terminal reward """

	train = model.train
	par = model.par
	dtype = train.dtype
	device = train.device

	m_pd = states_pd[...,0]

	if par.bequest_1 != 0.0:
		bequest_neq = par.bequest_1 * (m_pd + par.bequest_2)**(1-par.gamma) / (1-par.gamma)
		bequest_pos = par.bequest_1 * (m_pd * (1-par.tau_bequest) + par.bequest_2)**(1-par.gamma) / (1-par.gamma)
		bequest_util = torch.where(m_pd > 0, bequest_pos, bequest_neq)
	else:
		bequest_util = torch.zeros_like(m_pd)
	
	value_pd = bequest_util.unsqueeze(-1)
	return value_pd 

##############
# transition #
##############


def state_trans_pd(model,states,actions,outcomes,t0=0,t=None):
	""" transition to post-decision state """

	par = model.par

	# a. unpack
	m = states[...,0]
	wealth = m
	z1 = states[...,1]
	le = states[...,2]
	types = states[...,3:]

	# b. consumption
	c = outcomes[...,0]

	# c. post-decision
	m_pd = wealth-c
	z1_pd = z1
	le_pd = le

	# c. finalize
	states_pd = torch.cat((m_pd[...,None],z1_pd[...,None],le_pd[...,None],types),dim=-1)
	
	return states_pd 


def unemployment_prob(par, z_plus, t):
	""" unemployment probability """

	xi = par.a_p_unemp + par.b_p_unemp * (t+1) + par.c_p_unemp * z_plus + par.d_p_unemp * (t+1) * z_plus
	unemployment_prob = 1 / (1+torch.exp(-xi))

	return unemployment_prob


def retirement_income(le, AE, par):
	""" retirement income """

	ratio = le / AE

	# Define the masks
	mask1 = ratio < 0.23
	mask2 = (ratio >= 0.23) & (ratio < 1.38)
	# mask3 = ratio >= 1.38

	# Compute each branch
	income1 = 0.9 * le
	income2 = 0.2 * le + 0.32 * (le - 0.23 * AE)
	income3 = 0.57 * AE + 0.15 * (le - 1.38 * AE)

	# Combine using torch.where
	income = torch.where(mask1, income1, torch.where(mask2, income2, income3))

	return income



	return income
def state_trans(model,states_pd,shocks,t=None):
	""" state transition from post-decision to next period state """

	# a. unpack
	par = model.par
	train = model.train

	# states
	m_pd = states_pd[...,0]
	z_pd = states_pd[...,1]
	le_pd = states_pd[...,2]
	types = states_pd[...,3:]

	# shocks
	eta_base = shocks[...,0]
	epsilon_base = shocks[...,1]
	mix_eta = shocks[...,2]
	mix_epsilon = shocks[...,3]
	unemp_ = shocks[...,4]

	# b. adjust shape and scale quadrature nodes (when solving)
	if t is None:

		T,N = states_pd.shape[:-1]

		# states
		m_pd = expand_to_quad(m_pd,train.Nquad)
		z_pd = expand_to_quad(z_pd,train.Nquad)
		le_pd = expand_to_quad(le_pd,train.Nquad)
		types = expand_to_quad(types,train.Nquad)
		
		# types
		alpha_types = expand_to_states(par.alpha_types,states_pd)
		alpha_types = expand_to_quad(alpha_types, train.Nquad)
		theta_types = expand_to_states(par.theta_types,states_pd)
		theta_types = expand_to_quad(theta_types, train.Nquad)

		if par.use_reg:
			reg_constant_ = expand_to_states(par.reg_coefs[:,0],states_pd)
			reg_constant_ = expand_to_quad(reg_constant_, train.Nquad)
			reg_linear_ = expand_to_states(par.reg_coefs[:,1],states_pd)
			reg_linear_ = expand_to_quad(reg_linear_, train.Nquad)

		# shocks
		if train.do_quad:
			eta_base = expand_to_states(eta_base,states_pd)
			epsilon_base = expand_to_states(epsilon_base,states_pd)
			mix_eta = expand_to_states(mix_eta,states_pd)
			unemp = expand_to_states(unemp_,states_pd)
			mix_epsilon = expand_to_states(mix_epsilon,states_pd)
			
	else:
		alpha_types = par.alpha_types[None,:]
		theta_types = par.theta_types[None,:]
		if par.use_reg:
			reg_constant_ = par.reg_coefs[:,0][None,:]
			reg_linear_ = par.reg_coefs[:,1][None,:]

	
	# c. adjust persistent and transitory shocks
	sigma_eta = par.sigma_eta_vec[0] * (1-mix_eta) + par.sigma_eta_vec[1] * mix_eta
	mu_eta = par.mu_eta_vec[0] * (1-mix_eta) + par.mu_eta_vec[1] * mix_eta
	if t is None:
		eta = mu_eta + sigma_eta * eta_base
	else:
		eta = mu_eta + sigma_eta * eta_base
	
	sigma_epsilon = par.sigma_epsilon_vec[0] * (1-mix_epsilon) + par.sigma_epsilon_vec[1] * mix_epsilon
	mu_epsilon = par.mu_epsilon_vec[0] * (1-mix_epsilon) + par.mu_epsilon_vec[1] * mix_epsilon
	if t is None:
		epsilon = mu_epsilon + sigma_epsilon * epsilon_base
	else:
		epsilon = mu_epsilon + sigma_epsilon * epsilon_base

	# d. persistent income
	if t is None:
		z_plus_bef = par.rho_z * z_pd[:par.T_retired] + eta[:par.T_retired]
		z_plus_aft = z_pd[par.T_retired:]
		z_plus = torch.cat((z_plus_bef,z_plus_aft),dim=0)
	else:
		if t < par.T_retired:
			z_plus = par.rho_z * z_pd + eta
		else:
			z_plus = z_pd

	
	
	# e. types
	alpha =  torch.sum(types * alpha_types, dim=-1) 
	theta =  torch.sum(types * theta_types, dim=-1) 
	if par.use_reg:
		reg_constant = torch.sum(types * reg_constant_, dim=-1) 
		reg_linear = torch.sum(types * reg_linear_, dim=-1) 

	# f. compute income
	if t is None: # when solving

		# income trends
		g = par.g[1:].reshape(par.T-1,1,1) *  torch.ones_like(z_plus)
		t_mult = ((torch.arange(z_pd.shape[0],device=train.device)+1+1)/10).reshape(z_pd.shape[0],1,1) * torch.ones_like(z_plus)
		t_range = ((torch.arange(z_pd.shape[0],device=train.device)+1+1)).reshape(z_pd.shape[0],1,1) * torch.ones_like(z_plus)
		theta_ = theta * t_mult


		if not train.do_quad: # monte carlo expectations
			xi = par.a_p_unemp + par.b_p_unemp * t_mult + par.c_p_unemp * z_plus + par.d_p_unemp * t_mult * z_plus
			unemployment_prob = 1 / (1+torch.exp(-xi))
			indicator = unemployment_prob > unemp_ 
			unemp = indicator.float()
		
		# labor earnings
		y_plus_working_ = torch.exp(g[:par.T_retired] + alpha[:par.T_retired] + theta_[:par.T_retired] + z_plus_bef + epsilon[:par.T_retired])
		y_plus_working = (1-unemp[:par.T_retired]) * y_plus_working_
		if not par.use_reg:
			y_retired = retirement_income(le_pd[par.T_retired:], par.AE, par) * torch.ones_like(z_plus[par.T_retired:])
		else:
			le_predicted = reg_constant[par.T_retired:] + reg_linear[par.T_retired:] * z_plus[par.T_retired:]
			y_retired = retirement_income(le_predicted, par.AE, par) * torch.ones_like(z_plus[par.T_retired:])
		y_plus = torch.cat((y_plus_working,y_retired),dim=0)

		# lifetime earnings
		if not par.use_reg:
			# le_plus_working = le_pd[:par.T_retired] + y_plus_working / par.T_retired
			le_plus_working = le_pd[:par.T_retired] * (t_range[:par.T_retired] - 1) / t_range[:par.T_retired] + y_plus_working / t_range[:par.T_retired]
		else:
			le_plus_working = le_pd[:par.T_retired]
		
		# combine lifetime earnings
		le_plus_retired = le_pd[par.T_retired:] * torch.ones_like(z_plus[par.T_retired:])
		le_plus = torch.cat((le_plus_working,le_plus_retired),dim=0)

	else: # when simulating
		if t < par.T_retired: # working age

			# unemployment probability
			xi = par.a_p_unemp + par.b_p_unemp * (t+1+1)/10 + par.c_p_unemp * z_plus + par.d_p_unemp * (t+1+1)/10 * z_plus
			unemployment_prob = 1 / (1+torch.exp(-xi))
			indicator = unemployment_prob > unemp_ 
			unemp = indicator.float()
			y_plus = torch.exp(par.g[t+1] + alpha + theta*(t+1+1)/10 + z_plus + epsilon) * (1-unemp)

			# lifetime earnings
			if not par.use_reg:
				le_plus = le_pd / (t+2) * (t+1) + y_plus / (t+2)
			else:
				le_plus = le_pd * torch.ones_like(z_plus)

		else: # retired age
			# income
			if not par.use_reg:
				y_plus = retirement_income(le_pd, par.AE, par)  * torch.ones_like(z_plus)
			else:
				le_predicted = reg_constant  + reg_linear * z_pd
				y_plus = retirement_income(le_predicted, par.AE, par)  * torch.ones_like(z_plus)

			le_plus = le_pd * torch.ones_like(z_plus)
			
	# g. final income
	income = par.lambdaa * torch.clamp(y_plus,min=par.Y_min)**(1-par.tau)


	if t is not None and states_pd.shape[0] == model.sim.N:
		model.sim.income[t+1] =  income
		

	# h. cash-on-hand
	m_plus = par.R*m_pd + income # shape = (T,N,Nquad) or (T,N)
	
	# I. finalize

	states_plus = torch.cat((m_plus[...,None],z_plus[...,None],le_plus[...,None],types),dim=-1)
	return states_plus



def exploration(model,states,actions,eps,t=None):
	""" exploration for actions """


	actions_explored = actions
	actions_explored[...,0] = actions[...,0] + eps[...,0]

	return actions_explored


