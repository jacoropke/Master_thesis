import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' # without this python may crash when plotting from matplotlib
from copy import deepcopy
import numpy as np
import torch
torch.set_warn_always(True)

# local
import model_funcs
from EconDLSolvers import DLSolverClass, torch_uniform, compute_transfer

# class
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' # without this python may crash when plotting from matplotlib
from copy import deepcopy
import numpy as np
import torch
torch.set_warn_always(True)

from consav.quadrature import log_normal_gauss_hermite, gauss_hermite, normal_gauss_hermite

# local
import model_funcs
from EconDLSolvers import DLSolverClass, torch_uniform, compute_transfer

# class
class BufferStockModelClass(DLSolverClass):
	
	#########
	# setup #
	#########

	def setup(self,full=None):
		""" choose parameters """

		par = self.par
		sim = self.sim
		
		## Some commented out code is included such that you can easily disable certain features
		
		par.full = full if not full is None else torch.cuda.is_available()
		par.seed = 2 # seed for random number generator in torch

		# Time
		par.T = 60 # number of periods
		par.T_retired = 36-1 # number of periods retired

		# preferences
		par.beta = 0.960 # discount factor
		par.bequest_1 = 1.473 # bequest coefficint
		par.bequest_2 = 5.95 # minimum bequest utility parameter
		par.gamma = 2.0 # CRRA parameter

		# deterministic income polynomial
		par.omega_0 = 2.581 # constant
		par.omega_1 = 0.812 # linear
		par.omega_2 = -0.185 # quadratic

		## No deterministic age-trend parameters
		# par.omega_0 = 0.0 # constant
		# par.omega_1 = 0.0 # linear
		# par.omega_2 = 0.0 # quadratic


		# public sector
		par.tau = 0.185 # tax rate parameter
		# par.tau = 0.0 # tax rate parameter
		par.lambdaa = 1.763 # income scaling parameter
		par.Y_min = 3.0
		par.le_scaler = 1.0
		par.tau_bequest = 0.4

		# financial markets
		par.R = 1.03 # gross return
		par.borr_tightness = 0.5 # borrowing constraint tightness

		# transitory income shock
		par.p_epsilon = 0.13 # probability of using normal 1 
		par.mu_epsilon_1 = 0.271 # mean of normal 1

		par.sigma_epsilon_1 = 0.285 # std of normal 1
		par.sigma_epsilon_2 = 0.037 # std of normal 2
		par.Nepsilon = 4 # number of nodes for epsilon

		## No transitory income shock parameters
		# par.sigma_epsilon_1 = 0.0 # std of normal 1
		# par.sigma_epsilon_2 = 0.0 # std of normal 2
		# par.mu_epsilon_1 = 0.0 # mean of normal 1
		# par.Nepsilon = 2 # number of nodes for epsilon


		# persistent income shock
		par.rho_z = 0.959 # persistence
		par.mu_eta_1 = -0.085 # mean of normal 1
		par.p_z  = 0.407 # probability of using normal 1
		par.sigma_eta_1 = 0.364 # std of normal 1 
		par.sigma_eta_2 = 0.069 # std of normal 2
		## No persistent income shock parameteres
		# par.sigma_eta_1 = 0.0 # std of normal 1 
		# par.sigma_eta_2 = 0.0 # std of normal 2
		# par.mu_eta_1 = -0.0 # mean of normal 1
		par.Neta = 4 # number of nodes for eta
		par.disable_z_income = False # disable z income shock	


		#  unemployment shock
		par.Nunemp = 1 # number of nodes for unemployment
		par.a_p_unemp = -3.353
		par.b_p_unemp = -0.859
		par.c_p_unemp = -5.034
		par.d_p_unemp = -2.895

		## No unemployment shock parameters
		# par.a_p_unemp = -50.5 # unemp prob constant
		# par.b_p_unemp = -0.00 # unemp prob linear term in time
		# par.c_p_unemp = -0.0 # unemp prob linear term in permanent income state
		# par.d_p_unemp = -0.0 # unemp prob interaction term

		# fixed types
		par.Nalpha_types = 5
		par.Ntheta_types = 3
		par.sigma_alpha = 0.3
		par.sigma_theta = 0.196
		par.corr_alpha_theta = 0.768
		## No ex-ante heterogeneity beyond initial state variation
		# par.Nalpha_types = 1
		# par.Ntheta_types = 1
		# par.sigma_alpha = 0.0
		# par.sigma_theta = 0.0
		# par.corr_alpha_theta = 0.0
		par.set_type_probs = False
		par.type_probs_assign = np.zeros((par.Nalpha_types * par.Ntheta_types),dtype=np.float64)


		# initial permanent income
		par.mu_z1_0 = 0.0 # initial durable, mean
		par.sigma_z1_0 = 0.714 # initial durable, std=


		# estimated average earnings
		par.AE = 42.93581874649259
		par.use_reg = False # use regression to estimate le



		# states and shocks
		par.Nstates = 3 + par.Nalpha_types * par.Ntheta_types
		par.Nstates_pd = 3 + par.Nalpha_types * par.Ntheta_types
		par.Nshocks = 5 # number of shocks

		# outcomes and actions
		par.Noutcomes = 2 # number of outcomes
		par.KKT = False # use KKT conditions (for DeepFOC)
		par.NDC = 0 # number of discrete choices

		# scaling
		par.m_scaler = 1/10.0
		par.p_scaler = 1/5.0

		# policy prediction
		par.policy_predict = 'savings_rate' # 'savings_rate' or 'consumption'


		# c. simulation 
		sim.N = 100_000 # number of agentss

		par.death = True
		# par.death = False



	def allocate(self):
		""" allocate arrays  """

		# unpack
		par = self.par
		sim = self.sim
		train = self.train

		dtype = train.dtype
		device = train.device

		

		# Deterministic life cycle income
		par.g = torch.zeros(par.T,dtype=dtype,device=device)	# Martin - kappa
	
		for t in range(0,par.T):
			par.g[t] = par.omega_0 + par.omega_1*((t+1)/10) + par.omega_2*((t+1)/10)**2

		# remaining income parameters
		par.mu_eta_2 = - par.mu_eta_1 * par.p_z / (1-par.p_z) # mean of normal 2 - ensure zero mean
		par.mu_epsilon_2 = -par.mu_epsilon_1 * par.p_epsilon / (1-par.p_epsilon) # mean of normal 2 - ensure zero mean

		# actions
		par.Nactions = 1

		# quadrature transitory shock
		par.epsilon, par.epsilon_w = normal_gauss_hermite(1.0, par.Nepsilon)

		# quadrature persistent shock
		par.eta, par.eta_w = normal_gauss_hermite(1.0, par.Neta)

		# quadrature unemployment
		par.unemp = np.array([1.0, 0.0])
		par.unemp_w = np.array([0.0, 1.0])

		# borrowing constraint tightness
		par.a_low = torch.zeros(par.T,dtype=dtype,device=device) # borrowing constraint tightness
		# loop backwards to get borrowing constraint
		for t in range(par.T-2,-1,-1):
			par.a_low[t] = (par.borr_tightness/par.R) * (par.a_low[t+1] - par.lambdaa*par.Y_min**(1-par.tau))

		# convenient arrays
		par.p_z_vec = torch.tensor([par.p_z,1-par.p_z],dtype=dtype,device=device)
		par.mix_eta = torch.tensor([0,1],dtype=dtype,device=device)
		par.p_epsilon_vec = torch.tensor([par.p_epsilon,1-par.p_epsilon],dtype=dtype,device=device)
		par.mix_epsilon = torch.tensor([0,1],dtype=dtype,device=device)
		par.sigma_eta_vec = torch.tensor([par.sigma_eta_1,par.sigma_eta_2],dtype=dtype,device=device)
		par.sigma_epsilon_vec = torch.tensor([par.sigma_epsilon_1,par.sigma_epsilon_2],dtype=dtype,device=device)
		par.mu_eta_vec = torch.tensor([par.mu_eta_1,par.mu_eta_2],dtype=dtype,device=device)
		par.mu_epsilon_vec = torch.tensor([par.mu_epsilon_1,par.mu_epsilon_2],dtype=dtype,device=device)
		
		# gauss hermite for alpha, theta
		par.Ntypes = par.Nalpha_types * par.Ntheta_types
		gauss_1_node, gauss_1_weight = gauss_hermite(par.Nalpha_types)
		gauss_2_node, gauss_2_weight = gauss_hermite(par.Ntheta_types)

		## transform weights
		alpha_weight = gauss_1_weight / np.sqrt(np.pi)
		theta_weight = gauss_2_weight / np.sqrt(np.pi)

		## transform nodes
		alpha_node = 0 + gauss_1_node * par.sigma_alpha * np.sqrt(2)
		alpha_node = alpha_node[:,None] * np.ones((par.Nalpha_types,par.Ntheta_types))
		theta_node = 0 + np.sqrt(2) *  par.sigma_theta * (par.corr_alpha_theta * gauss_1_node[:,None] + np.sqrt(1 - par.corr_alpha_theta**2) * gauss_2_node[None,:])
		par.alpha_types = alpha_node.flatten()
		par.theta_types = theta_node.flatten()
		if not par.set_type_probs:
			par.type_probs = alpha_weight[:,None] * theta_weight[None,:]
			par.type_probs = par.type_probs.flatten()
		else:
			par.type_probs = par.type_probs_assign

		# convert to torch
		par.epsilon = torch.tensor(par.epsilon,dtype=dtype,device=device)
		par.epsilon_w = torch.tensor(par.epsilon_w,dtype=dtype,device=device)
		par.eta = torch.tensor(par.eta,dtype=dtype,device=device)
		par.eta_w = torch.tensor(par.eta_w,dtype=dtype,device=device)
		par.unemp = torch.tensor(par.unemp,dtype=dtype,device=device)
		par.unemp_w = torch.tensor(par.unemp_w,dtype=dtype,device=device)
		par.alpha_types = torch.tensor(par.alpha_types,dtype=dtype,device=device)
		par.theta_types = torch.tensor(par.theta_types,dtype=dtype,device=device)
		par.type_probs = torch.tensor(par.type_probs,dtype=dtype,device=device)
		par.Y_min_tensor = torch.tensor([par.Y_min],dtype=dtype,device=device)

		# d. scaling vector - not used
		par.scale_vec_states = torch.tensor([par.m_scaler,par.p_scaler],dtype=dtype,device=device)
		par.scale_vec_states_pd = torch.tensor([par.m_scaler,par.p_scaler],dtype=dtype,device=device)


		# e. simulation		
		sim.states = torch.zeros((par.T,sim.N,par.Nstates),dtype=dtype,device=device)
		sim.states_pd = torch.zeros((par.T,sim.N,par.Nstates_pd),dtype=dtype,device=device)
		sim.shocks = torch.zeros((par.T,sim.N,par.Nshocks),dtype=dtype,device=device)
		sim.outcomes = torch.zeros((par.T,sim.N,par.Noutcomes),dtype=dtype,device=device)
		sim.actions = torch.zeros((par.T,sim.N,par.Nactions),dtype=dtype,device=device)
		sim.reward = torch.zeros((par.T,sim.N),dtype=dtype,device=device)
		sim.income = torch.zeros((par.T,sim.N),dtype=dtype,device=device)
		sim.unemployment_value_0 = torch.zeros((sim.N),dtype=dtype,device=device)
		sim.working_income_after_unemp = np.zeros((par.T,sim.N))
		sim.working_income_before_unemp = np.zeros((par.T,sim.N))

		sim.euler_error = torch.zeros((par.T-1, sim.N),dtype=dtype,device=device)
		sim.R = 0.0
		par.survival_probs = np.array([
			0.998037,  # Age 25
			0.997918,  # Age 26
			0.997798,  # Age 27
			0.997670,  # Age 28
			0.997543,  # Age 29
			0.997426,  # Age 30
			0.997317,  # Age 31
			0.997213,  # Age 32
			0.997119,  # Age 33
			0.997026,  # Age 34
			0.996926,  # Age 35
			0.996825,  # Age 36
			0.996705,  # Age 37
			0.996556,  # Age 38
			0.996392,  # Age 39
			0.996220,  # Age 40
			0.996042,  # Age 41
			0.995856,  # Age 42
			0.995663,  # Age 43
			0.995460,  # Age 44
			0.995226,  # Age 45
			0.994936,  # Age 46
			0.994636,  # Age 47
			0.994309,  # Age 48
			0.993963,  # Age 49
			0.993585,  # Age 50
			0.993176,  # Age 51
			0.992736,  # Age 52
			0.992264,  # Age 53
			0.991760,  # Age 54
			0.991224,  # Age 55
			0.990656,  # Age 56
			0.990056,  # Age 57
			0.989423,  # Age 58
			0.988758,  # Age 59
			0.988060,  # Age 60
			0.987329,  # Age 61
			0.986565,  # Age 62
			0.985768,  # Age 63
			0.984938,  # Age 64
			0.984075,  # Age 65
			0.983179,  # Age 66
			0.982250,  # Age 67
			0.981287,  # Age 68
			0.980291,  # Age 69
			0.979261,  # Age 70
			0.978198,  # Age 71
			0.977101,  # Age 72
			0.975970,  # Age 73
			0.974806,  # Age 74
			0.973608,  # Age 75
			0.972376,  # Age 76
			0.971110,  # Age 77
			0.969810,  # Age 78
			0.968476,  # Age 79
			0.967108,  # Age 80
			0.965706,  # Age 81
			0.964270,  # Age 82
			0.962800,  # Age 83
			0.961296,  # Age 84
			0.000000   # Age 85 (certain death)
		])


		if par.death:
			par.s_p = par.survival_probs[0:par.T]
			par.s_p[-1] = 0.0 # set last period to zero
		else:
			par.s_p = np.ones(par.T)
		par.s_p[-1] = 0.0 # set last period to zero

		par.s_p_torch = torch.tensor(par.s_p,dtype=dtype,device=device)

		par.uncon_survival = np.zeros(par.T)
		par.uncon_survival[0] = 1.0
		for t in range(1,par.T):
			par.uncon_survival[t] = par.uncon_survival[t-1] * par.s_p[t-1]


		par.reg_coefs = np.array([
								[  5.27180752,   1.97159299],
        						[  6.70816648,   3.1305626 ],
        						[  9.15380729,   4.16292218],
        						[ 10.56030159,   5.41086065],
        						[ 15.42453819,   8.42673469],
        						[ 23.43150292,  14.01005501],
        						[ 22.18113694,  11.99934688],
        						[ 33.87235899,  19.93866987],
        						[ 53.22907325,  34.91104643],
        						[ 49.80426361,  29.27348829],
        						[ 78.05442847,  50.46252589],
        						[123.57757852,  84.66704312],
        						[119.37137305,  85.58388724],
        						[204.24641581, 151.43075307],
        						[335.87848625, 235.55655954]
								])
			
		par.reg_coefs = torch.tensor(par.reg_coefs, dtype=dtype, device=device)



	#########
	# train #
	#########

	def setup_train(self):
		""" default parameters for training """
		
		par = self.par
		train = self.train
		dtype = train.dtype
		device = train.device

		# a. neural network

		if par.policy_predict == 'savings_rate':

			train.policy_activation_final = ['sigmoid']
			train.min_actions = torch.tensor([0.0],dtype=dtype,device=device)
			train.max_actions = torch.tensor([0.9999],dtype=dtype,device=device)
		
		elif par.policy_predict == 'consumption':
			train.policy_activation_final = ['softplus']
			train.min_actions = torch.tensor([1e-8],dtype=dtype,device=device)
			train.max_actions = torch.tensor([10000.0],dtype=dtype,device=device)

		else:

			raise ValueError('policy_predict must be either savings_rate or consumption')
		
		# b. misc
		train.terminal_actions_known = True # use terminal actions
		train.use_input_scaling = False # use input scaling

		# c. algorithm specific
	
		# DeepSimulate

		if train.algoname != 'DeepSimulate':
			if par.policy_predict == 'savings_rate':

				# train.epsilon_sigma = np.array([0.2])
				train.epsilon_sigma = np.array([0.1])
				train.epsilon_sigma_min = np.array([0.0])
			else:
				train.epsilon_sigma = np.array([0.2])
				train.epsilon_sigma_min = np.array([0.0])
						
		train.Nneurons_policy = np.array([600,600])
		if train.algoname == 'DeepSimulate':
			train.Nneurons_policy = np.array([600,600])

		if train.algoname == 'DeepVPD':
			train.Nneurons_value = np.array([400,400])
			train.N_value_NN = 3


		if train.algoname == 'SimEGM':
			train.N = 250
			train.batch_size= train.N
			train.learning_rate_policy_decay = 0.9998
			train.tau = 0.2
		train.Nmc = 400 
		train.update_quad_w = True # update quadrature weights (necessary for unemployment probability)
		train.do_quad = True
		train.new_mc = True
		train.update_mc_freq = 10


	def allocate_train(self):
		""" allocate memory training """

		par = self.par
		train = self.train
		dtype = train.dtype
		device = train.device

		# a. dependent settings
		if train.algoname == 'DeepFOC':
			train.eq_w = train.eq_w / torch.sum(train.eq_w) # normalize

		# b. simulation
		train.states = torch.zeros((par.T,train.N,par.Nstates),dtype=dtype,device=device)
		train.states_pd = torch.zeros((par.T,train.N,par.Nstates_pd),dtype=dtype,device=device)
		train.shocks = torch.zeros((par.T,train.N,par.Nshocks),dtype=dtype,device=device)
		train.outcomes = torch.zeros((par.T,train.N,par.Noutcomes),dtype=dtype,device=device)
		train.actions = torch.zeros((par.T,train.N,par.Nactions),dtype=dtype,device=device)
		train.reward = torch.zeros((par.T,train.N),dtype=dtype,device=device)
		train.income = torch.zeros((par.T,train.N),dtype=dtype,device=device)
	
	def quad(self):
		""" quadrature nodes and weights """

		par = self.par
		train = self.train
		dtype = train.dtype
		device = train.device

		if train.do_quad:

			par = self.par

			# Reorder meshgrid so unemployment is last
			eta, epsilon, mix_eta, mix_epsilon, unemployment = torch.meshgrid(
				par.eta, par.epsilon, par.mix_eta, par.mix_epsilon, par.unemp, indexing='ij'
			)
			eta_w, epsilon_w, p_mixeta, p_mix_epsilon, unemp_w = torch.meshgrid(
				par.eta_w, par.epsilon_w, par.p_z_vec, par.p_epsilon_vec, par.unemp_w, indexing='ij'
			)

			# Stack and flatten with unemployment last
			quad = torch.stack((
				eta.flatten(), 
				epsilon.flatten(), 
				mix_eta.flatten(), 
				mix_epsilon.flatten(), 
				unemployment.flatten()
			), dim=-1)

			# Corresponding weights
			quad_w = (
				eta_w.flatten() *
				epsilon_w.flatten() *
				p_mixeta.flatten() *
				p_mix_epsilon.flatten() *
				unemp_w.flatten()
			)

			return quad, quad_w

		else:


			# a. antithetic normal draws
			eta_ = torch.normal(0.0, 1.0, size=(par.T-1, train.N, train.Nmc//2))
			eta = torch.cat((eta_, -eta_), dim=-1)

			epsilon_ = torch.normal(0.0, 1.0, size=(par.T-1, train.N, train.Nmc//2))
			epsilon = torch.cat((epsilon_, -epsilon_), dim=-1)

			# b. antithetic uniform draws for Bernoulli sampling

			# For unemployment (uniform [0,1] draws, e.g. for quantile thresholds)
			unemployment_u = torch.rand(size=(par.T-1, train.N, train.Nmc//2))
			unemployment = torch.cat((unemployment_u, 1 - unemployment_u), dim=-1)

			# For mix_eta ~ Bernoulli(1 - p_z)
			u_mix_eta = torch.rand(size=(par.T-1, train.N, train.Nmc//2))
			mix_eta_ = (u_mix_eta < (1 - par.p_z)).float()
			mix_eta_antithetic = ((1 - u_mix_eta) < (1 - par.p_z)).float()
			mix_eta = torch.cat((mix_eta_, mix_eta_antithetic), dim=-1)

			# For mix_epsilon ~ Bernoulli(1 - p_epsilon)
			u_mix_epsilon = torch.rand(size=(par.T-1, train.N, train.Nmc//2))
			mix_epsilon_ = (u_mix_epsilon < (1 - par.p_epsilon)).float()
			mix_epsilon_antithetic = ((1 - u_mix_epsilon) < (1 - par.p_epsilon)).float()
			mix_epsilon = torch.cat((mix_epsilon_, mix_epsilon_antithetic), dim=-1)

			# Final stack
			mc_draws = torch.stack((eta, epsilon, mix_eta, mix_epsilon, unemployment), dim=-1)  # shape (T-1, N, Nmc, 5)
			mc_w = torch.ones((train.Nmc,)) / train.Nmc  # shape (Nmc,)

			return mc_draws, mc_w


	def quad_w_update(self, states_plus):
		"""Create quad weights with unemployment dimension LAST"""
		par = self.par
		train = self.train

		# a. Unpack dimensions
		T_dim = states_plus.shape[0]  # T-1
		N_dim = states_plus.shape[1]  # N
		z_plus = states_plus[..., 1]  # [T-1, N, Nquad]

		# b. Reshape with unemployment LAST [T-1, N, Neta, Nepsilon, 2, 2, 2]
		z_plus_reshaped = z_plus.view(T_dim, N_dim, par.Neta, par.Nepsilon, 2, 2, 2)

		# c. Compute xi (same calculation, different dimension order)
		t_range = (torch.arange(T_dim, device=z_plus.device)[:, None, None, None, None, None, None] + 1 +1) / 10
		xi = (
			par.a_p_unemp + 
			par.b_p_unemp * t_range + 
			par.c_p_unemp * z_plus_reshaped + 
			par.d_p_unemp * t_range * z_plus_reshaped
		)  # [T-1,N,Neta,Nepsilon,2,2,2]

		# d. Compute probabilities (unemployment dimension last)
		unemp_w = torch.zeros_like(xi)
		unemp_w[..., 0] = 1 / (1 + torch.exp(-xi[..., 0]))  # P(unemployed)
		unemp_w[..., 1] = 1 - unemp_w[..., 0]           # P(employed)

		# e. Create partial quadrature weights (excluding unemployment)
		weights = torch.einsum(
			'a,b,c,d->abcd',
			par.eta_w, 
			par.epsilon_w,
			par.p_z_vec,
			par.p_epsilon_vec
		)  # [Neta, Nepsilon, 2, 2]

		# f. Combine all weights
		weights_expanded=  weights[None,None,:,:,:,:,None].expand(T_dim,N_dim,par.Neta,par.Nepsilon,2,2,2)
		quad_w = (weights_expanded * unemp_w).reshape(T_dim, N_dim, -1)

		return quad_w
		
	#########
	# draw #
	#########

	def draw_initial_states(self,N,training=False, shocks=None):
		""" draw initial state (m,p,t) """

		par = self.par

		
		# a. draw permanent income
		if training and self.train.algoname != 'DeepSimulate':
			sigma_z1 = par.sigma_z1_0 * 1.0
		else:
			sigma_z1 = par.sigma_z1_0
		z_0 = torch.normal(0.0,1.0,size=(N,))
		z_0 = 0.0 + sigma_z1 * z_0

		# b. types
		type_index = torch.multinomial(par.type_probs.flatten().cpu(), N, replacement=True,generator=self.torch_gen)
		type_dummy = torch.zeros((N,par.Ntypes),dtype=torch.float32)
		
		type_dummy[torch.arange(N),type_index] = 1.0


		# c. initial wealth and lifetime earnings
		#  simulate period 0 income
		## i. unemployment
		unemployment = shocks[...,4]
		xi = par.a_p_unemp + par.b_p_unemp * ((0+1)/10) + par.c_p_unemp * z_0 + par.d_p_unemp * z_0 * ((0+1)/10)
		unemp_prob = 1/(1+np.exp(-xi))
		unemployment_indicator = unemp_prob > unemployment.to('cpu')
		unemployment_value = unemployment_indicator.float()
		if not training:
			self.sim.unemployment_value_0 = unemployment_value

		## ii. types
		alpha = par.alpha_types[type_index]
		theta = par.theta_types[type_index]

		## iii. transitory income shock
		epsilon_base = shocks[...,1]
		mix_epsilon = shocks[...,3]
		sigma_epsilon = (1-mix_epsilon) * par.sigma_epsilon_1 + mix_epsilon * par.sigma_epsilon_2
		mu_epsilon = (1-mix_epsilon) * par.mu_epsilon_1 + mix_epsilon * par.mu_epsilon_2
		epsilon = mu_epsilon + sigma_epsilon * epsilon_base

		## iv. compute earnings and income
		if par.disable_z_income:
			z_multiplier = 0.0
		else:
			z_multiplier = 1.0
		earnings = np.exp(par.g[0].to('cpu') + alpha.to('cpu') + theta.to('cpu')*(1/10) + z_multiplier * z_0.to('cpu') + epsilon.to('cpu')) * (1-unemployment_value.to('cpu'))
		if not training:
			self.sim.working_income_after_unemp[0] = earnings
		earnings_ = np.clip(earnings, par.Y_min, 1000000.0)
		income = par.lambdaa * earnings_**(1-par.tau) # income in period 0		

		# v. set initial cast-on-hand as earnings
		m0 = income


		# set initial life time earnings
		if not par.use_reg:
			le_0 = earnings
		else:
			le_0 = torch.zeros_like(earnings)
		
		return torch.cat((m0[:,None],z_0[:,None],le_0[:,None],type_dummy),dim=-1)
		

	def draw_shocks(self,N):
		""" draw shocks """

		par = self.par

		# a. persistent income shocks
		eta = torch.normal(0,1.0,size=(par.T,N))

		# b. transitory income shocks
		epsilon = torch.normal(0,1.0,size=(par.T,N))

		# c. unemployment - uniform number for comparison
		unemployment = torch.rand(size=(par.T,N))
		unemployment = torch.rand(size=(par.T,N)) # MISTAKE BUT CHANGES RESULTS VERY SLIGHTLY DUE TO DIFFERENT DRAW FOR GIVEN SEED

		# d. mixture for eta
		mix_eta = torch.bernoulli((1-par.p_z)*torch.ones((par.T,N)))

		# e. mixture for xi1
		mix_epsilon = torch.bernoulli((1-par.p_epsilon)*torch.ones((par.T,N)))

		return torch.stack((eta,epsilon,mix_eta,mix_epsilon,unemployment),dim=-1)
	
	def draw_exploration_shocks(self,epsilon_sigma,N):
		""" draw exploration shockss """

		par = self.par

		eps = torch.zeros((par.T,N,par.Nactions))
		for i_a in range(par.Nactions):
			# eps[:,:,i_a] = torch.normal(0,epsilon_sigma[i_a],(par.T,N))
			eps[:,:,i_a] = torch.normal(0,epsilon_sigma[i_a],(par.T,N)).clamp(0.0,1.0)

		return eps

	def draw_exo_actions(self,N):
		""" draw exogenous actions """

		par = self.par

		exo_actions = torch_uniform(0.01,0.8,size=(par.T,N,par.Nactions))
	
		return exo_actions

	###################
	# model functions #
	###################

	outcomes = model_funcs.outcomes
	reward = model_funcs.reward
	discount_factor = model_funcs.discount_factor
	
	terminal_actions = model_funcs.terminal_actions
	terminal_reward_pd = model_funcs.terminal_reward_pd
		
	state_trans_pd = model_funcs.state_trans_pd
	state_trans = model_funcs.state_trans
	exploration = model_funcs.exploration
	marg_util = model_funcs.marg_util_c
	inv_marg_util = model_funcs.inverse_marg_util
	marg_bequest = model_funcs.marg_bequest

	def add_transfer(self,transfer):
		""" add transfer to initial states """

		par = self.par
		sim = self.sim

		sim.states[0,:,0] += transfer



	def compute_policy_on_grids(self):
		"""
		Compute policy on grids using a neural network, with states:
		- m: cash-on-hand
		- z: persistent income
		- life: lifetime earnings
		- fe: one-hot encoded fixed-effect index

		Output shape: (T-1, Ntypes, Nlife, Nz, Nm)
		"""
		# Unpack
		par = self.par
		train = self.train
		Ntypes = par.Nalpha_types * par.Ntheta_types

		# Grids
		m_grid_len = 250
		m_grids = np.zeros((par.T-1, m_grid_len))
		z_grid = np.linspace(-5.0, 5.0, 200)
		life_grid = np.linspace(1.0, 50.0, 100)
		fe_ids = np.arange(Ntypes)

		# Output container
		sol_con_grid = np.zeros((par.T-1, Ntypes, len(life_grid), len(z_grid), m_grid_len))
		sol_sr_grid = np.zeros((par.T-1, Ntypes, len(life_grid), len(z_grid), m_grid_len))

		# Loop over time and fixed effects
		for t in range(par.T-1):
			print(f't = {t}')

			# Meshgrid once
			m_grids[t] = np.linspace(par.a_low[t].cpu().numpy(), 100.0, m_grid_len)
			life, z, m = np.meshgrid(life_grid, z_grid, m_grids[t], indexing='ij')
			life_flat = life.reshape(-1)
			z_flat = z.reshape(-1)
			m_flat = m.reshape(-1)
			N = m_flat.shape[0]


			for i_fe in fe_ids:
				print(f'i_fe = {i_fe}')
				fe_onehot = np.zeros((N, Ntypes))
				fe_onehot[:, i_fe] = 1.0

				# State order for NN: [m, z, life, onehot...]
				states_grid = np.concatenate(
					(m_flat[:, None], z_flat[:, None], life_flat[:, None], fe_onehot), axis=1
				)

				states_grid_tensor = torch.tensor(states_grid, dtype=train.dtype, device=train.device)

				with torch.no_grad():
					output_saving = self.eval_policy(self.policy_NN, states_grid_tensor, t=t)


				output_np = self.outcomes(states_grid_tensor, output_saving, t=t).cpu().numpy()
				output_saving = output_saving.cpu().numpy()

				sol_con_grid[t, i_fe, :, :, :] = output_np[...,0].reshape(len(life_grid), len(z_grid), m_grid_len)
				sol_sr_grid[t, i_fe, :, :, :] = output_saving[...,0].reshape(len(life_grid), len(z_grid), m_grid_len)

		return sol_con_grid,sol_sr_grid,  m_grids, z_grid, life_grid




	def compute_euler_errors(self,Nbatch_share=0.01):
		""" compute euler error"""

		par = self.par
		sim = self.sim
		train = self.train

		Nbatch = int(Nbatch_share*sim.N)

		for i in range(0,sim.N,Nbatch):

			index_start = i
			index_end = i + Nbatch

			with torch.no_grad():
				
				# a. get consumption and states today
				c = sim.outcomes[:par.T-1,index_start:index_end,0]
				states = sim.states[:par.T-1,index_start:index_end]
				actions = sim.actions[:par.T-1,index_start:index_end]
				outcomes = sim.outcomes[:par.T-1,index_start:index_end]

				# b. post-decision states
				states_pd = self.state_trans_pd(states,actions,outcomes)

				# c. next-period states
				states_next = self.state_trans(states_pd,train.quad)

				# d. next-period action
				actions_next_before = self.eval_policy(self.policy_NN,states_next[:par.T-2],t0=1)
				actions_next_after = self.terminal_actions(states_next[par.T-2])
				actions_next = torch.cat((actions_next_before,actions_next_after[None,...]),dim=0)

				# e. next-period consumption
				c_next = self.outcomes(states_next,actions_next,t0=1)[...,0]

				# f. marginal utility next period
				marg_util_next = self.marg_util(c_next, par)

				# g. expected marginal utility next period
				quad_w = self.quad_w_update(states_next)
				exp_marg_util_next = torch.sum(quad_w*marg_util_next, dim=-1)

				marg_bequest = self.marg_bequest(states_pd[...,0])

				q = par.s_p_torch[:-1, None] * par.R * exp_marg_util_next + (1 - par.s_p_torch[:-1, None]) * marg_bequest

				# h. euler error
				euler_error_Nbatch = self.inv_marg_util(par.beta*q,par) / c - 1
				euler_error_Nbatch = torch.abs(euler_error_Nbatch)
				sim.euler_error[:par.T-1,index_start:index_end] = euler_error_Nbatch