import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' # without this python may crash when plotting from matplotlib
from copy import deepcopy
from types import SimpleNamespace
import pickle
import numpy as np
import torch
torch.set_warn_always(True)

from EconModel import EconModelClass, jit
from consav.grids import nonlinspace

from BufferStockModel import BufferStockModelClass

# local
from egm import  simulate, simulate_income


class BufferStockModelEGMClass(EconModelClass,BufferStockModelClass):

    def settings(self):
        """ basic settings """
        
        self.namespaces = ['par','sim','egm'] # must be numba-able
        
        # save
        self.other_attrs = ['info','train'] # other attributes to save
        self.savefolder = 'saved' # folder for saved data

        # info
        self.info = {}

        # train
        self.train = SimpleNamespace()
        self.train.dtype = torch.float32
        self.train.device = 'cpu'

        # cpp
        self.cpp_filename = 'cppfuncs/egm.cpp'
        self.cpp_options = {'compiler':'vs'}	

    def setup(self):
        """ choose parameters """

        par = self.par
        egm = self.egm
        sim = self.sim

        # a. from BufferStockModelClass
        # self._setup_default()
        super().setup(full=True)
        self._setup_default()

        # b. egm
        egm.Nm_low = 350 # number of grid points
        egm.Nm_high = 20
        egm.Nm = egm.Nm_low  + egm.Nm_high # number of grid points
        
        # egm.Nm = 350 # number of grid points
        # egm.Nm = 650 # number of grid points
        # egm.Nm_pd = 350 # number of grid points
        egm.Nm_pd_low = 250 # number of grid points
        egm.Nm_pd_high = 20
        egm.Nm_pd = egm.Nm_pd_low + egm.Nm_pd_high # number of grid points

        # egm.Nm_pd = 650 # number of grid points
        # egm.Nz = 150 # number of grid points
        egm.Nz = 250 # number of grid points

        # egm.m_max_normal = 10000.0 # max cash-on-hand
        egm.m_max_normal = 800.0 # max cash-on-hand
        egm.m_max_normal = 1200.0 # max cash-on-hand
        egm.m_max_normal = 1500.0 # max cash-on-hand
        # egm.m_max_normal = 2000.0 # max cash-on-hand
        # egm.m_max_normal = 450.0 # max cash-on-hand
        # egm.m_max_normal = 900.0 # max cash-on-hand
        egm.m_max_high = 20000.0 # max cash-on-hand
        # egm.m_max_high = 22000.0 # max cash-on-hand

        egm.le_low = 0.0
        egm.le_high = 200.0
        egm.le_high = 300.0
        # egm.le_high = 600.0
        egm.le_very_high = 2500.0
        # egm.le_very_high = 5000.0
        egm.Nle = 50 # number of grid points for lifetime earnings
        egm.Nle = 100 # number of grid points for lifetime earnings
        egm.Nle = 150 # number of grid points for lifetime earnings
        egm.Nle = 250 # number of grid points for lifetime earnings
        # egm.Nle = 3 # number of grid points for lifetime earnings
        # egm.Nle = 350 # number of grid points for lifetime earnings
        # egm.Nle = 450 # number of grid points for lifetime earnings
        # egm.Nle = 550 # number of grid points for lifetime earnings
        # egm.Nle = 50 # number of grid points for lifetime earnings
        # egm.Nle = 50 # number of grid points for lifetime earnings
        
        egm.z1_max = 4.3 # max permanent income
        egm.z1_min = -4.3 # min permanent income

        sim.N = 100_000
        # sim.N = 20_000
        # sim.N = 200_000


        # b. get number of cores
        par.cppthreads = min(egm.Nz,os.cpu_count())

    def allocate(self):
        """ allocate arrays """

        par = self.par
        sim = self.sim
        egm = self.egm



        # a. from BufferStockModelClass
        super().allocate()
        if par.use_reg:
            egm.Nle = 1

        self.prepare_simulate_R()

        for k,v in par.__dict__.items():
            if isinstance(v,torch.Tensor): par.__dict__[k] = v.to(torch.float64).cpu().numpy()

        for k,v in sim.__dict__.items():
            if isinstance(v,torch.Tensor): sim.__dict__[k] = v.to(torch.float64).cpu().numpy()

        # b. grids
        self.create_EGM_grids()

        # c. R transfer
        sim.R_transfer = np.zeros(egm.Ntransfer)
        sim.R_transfers = np.zeros((sim.reps,egm.Ntransfer))


        # d. extra arrays
        sim.working_income_before_unemp = np.zeros((par.T,sim.N))
        sim.working_income_after_unemp = np.zeros((par.T,sim.N))
        sim.alpha = np.zeros((sim.N))
        sim.theta = np.zeros((sim.N))
        sim.eta = np.zeros((par.T,sim.N))
        sim.epsilon = np.zeros((par.T,sim.N))


        # decomposition
        sim.deterministic_income = np.zeros((par.T,sim.N))
        sim.fixed_effect_income_type = np.zeros((par.T,sim.N))
        sim.trend_income_type = np.zeros((par.T,sim.N))
        sim.transitory_income = np.zeros((par.T,sim.N))
        sim.persistent_income = np.zeros((par.T,sim.N))
        sim.unemployment = np.zeros((par.T,sim.N))
        sim.lifetime_earnings = np.zeros(sim.N)
        sim.lifetime_earnings_compare = np.zeros(sim.N)
        sim.years_employed = np.zeros(sim.N)
        sim.terminal_reward = np.zeros((par.T,sim.N))


        sim.euler_error = np.zeros((par.T,sim.N))
                
    def create_EGM_grids(self):
        """ create grids for EGM and dependent variables """

        par = self.par
        egm = self.egm

        # a. grids for dynamic states
        # egm.m_pd_grid = nonlinspace(0.0,egm.m_max_normal,egm.Nm_pd-1,1.5)
        # egm.m_pd_grid = np.concatenate((egm.m_pd_grid,np.array([egm.m_max_high])))
        # egm.m_grid = nonlinspace(0.0,egm.m_max_normal,egm.Nm-1,1.5)
        # egm.m_grid = np.concatenate((egm.m_grid,np.array([egm.m_max_high])))
        egm.m_pd_grids = np.zeros((par.T,egm.Nm_pd))
        egm.m_grids = np.zeros((par.T,egm.Nm))
        for t in range(par.T):

            # m_pd_grid = nonlinspace(par.a_low[t],egm.m_max_normal,egm.Nm_pd-1,1.4)
            m_pd_grid = nonlinspace(par.a_low[t],egm.m_max_normal,egm.Nm_pd_low,1.5)
            m_pd_grid_high = np.linspace(egm.m_max_normal,egm.m_max_high,egm.Nm_pd_high)
            egm.m_pd_grids[t] = np.concatenate((m_pd_grid,m_pd_grid_high))
            # m_grid = nonlinspace(par.a_low[t],egm.m_max_normal,egm.Nm-1,1.4)
            m_grid = nonlinspace(par.a_low[t],egm.m_max_normal,egm.Nm_low,1.5)
            m_grid_high = np.linspace(egm.m_max_normal,egm.m_max_high,egm.Nm_high)
            # egm.m_grids[t] = np.concatenate((m_grid,np.array([egm.m_max_high])))
            egm.m_grids[t] = np.concatenate((m_grid,m_grid_high))
        

        egm.z_grid = np.linspace(egm.z1_min,egm.z1_max,egm.Nz)
        
        
        # egm.le_grid = np.linspace(egm.le_low,egm.le_high,egm.Nle-1)
        # egm.le_grid = np.linspace(egm.le_low,egm.le_high,egm.Nle)
        if not par.use_reg:
            egm.le_grid = nonlinspace(egm.le_low,egm.le_high,egm.Nle-1,1.5)
            egm.le_grid = np.concatenate((egm.le_grid,np.array([egm.le_very_high])))
        else:
            egm.le_grid = np.array([0.0])


        # b. solution objects
        egm.sol_con = np.zeros((par.T,par.Ntypes,egm.Nle,egm.Nz,egm.Nm))
        

        # c. misc
        pos = np.array([1,5,10,25,50,100,500,1000])
        neg = np.array([1,5,10,25,50,100,500,1000,1500,2000,2500,3000,3500,4000,4500,5000])
        egm.transfer_grid = np.concatenate((-np.flip(neg),np.zeros(1),pos))/10_000
        egm.Ntransfer = egm.transfer_grid.size
            
    ############
    # simulate #
    ############

    def simulate_R(self,final=False):	
        """ simulate life time reward """

        par = self.par
        sim = self.sim

        # a. simulate
        with jit(self) as model:
            simulate(model.par,model.egm,model.sim,final=final)

        # b. compute R
        beta = par.beta 
        beta_t = np.zeros((par.T,sim.N))
        for t in range(par.T):
            beta_t[t] = beta**t
        
        cumulative_reward = np.zeros((par.T,sim.N))

        discounted_reward = 0.0
        for t in range(par.T):
            discounted_reward += beta_t[t] * (par.uncon_survival[t] *sim.reward[t] + par.uncon_survival[t]  * (1-par.s_p[t]) * sim.terminal_reward[t])
        
        sim.R = np.sum(discounted_reward) / sim.N

           
    def compute_transfer_func(self):
        """ compute EGM utility for different transfer levels"""

        sim = self.sim
        egm = self.egm

        R_transfer = np.zeros(egm.Ntransfer)
        for i, transfer in enumerate(self.egm.transfer_grid):
            
            sim_ = deepcopy(sim) # save
            
            sim.states[0,:,0] += transfer * sim.states[0,:,0] # as percent of cash-on-hand

            self.simulate_R()
            R_transfer[i] = sim.R
        
            sim = self.sim = sim_ # reset

        sim.R_transfer = R_transfer
    
    def simulate_income(self):
        """ simulate income """

        # a. simulate
        with jit(self) as model:
            simulate_income(model.par,model.egm, model.sim)
    

    ########
    # save #
    ########

    def save(self,filename):
        """ save the model """

        # a. create model dict
        model_dict = self.as_dict()

        # b. save to disc
        with open(f'{filename}', 'wb') as f:
            pickle.dump(model_dict, f)	
