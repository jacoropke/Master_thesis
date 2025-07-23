//////////////////////////
// 1. external includes //
//////////////////////////

// standard C++ libraries
#include <windows.h>
#include <stdio.h>
#include <cmath>
#include <omp.h>
#include <typeinfo>
#include <algorithm>

///////////////
// 2. macros //
///////////////

#define MAX(X,Y) ((X)>(Y)?(X):(Y))
#define MIN(X,Y) ((X)<(Y)?(X):(Y))
#define BOUND(X,A,B) MIN(MAX(X,A),B)

#define EXPORT extern "C" __declspec(dllexport)

/////////////////
// 3. includes //
/////////////////

#include "index.cpp"
#include "linear_interp.cpp"
#include "logs.cpp"

////////////////
// 4. structs //
////////////////

#include "par_struct.cpp"
#include "egm_struct.cpp"
#include "sim_struct.cpp"

/////////////
// 5. main //
/////////////

double marg_util_con(par_struct* par, double c){

    double mu = pow(c,-par->gamma);

    return mu;

} // marg_util_con

double inverse_marg_util_con(par_struct* par, double mu){

    double c = pow(mu,-1.0/par->gamma);

    return c;

} // inverse_marg_util_con


double compute_unemp_prob(par_struct* par, long long t, double z_plus){
    // compute probability of unemployment

    double xi = par->a_p_unemp + par->b_p_unemp * (t+1+1)/10 + par->c_p_unemp * z_plus + par->d_p_unemp * z_plus * (t+1+1)/10;

    double prob = 1.0 / (1.0 + exp(-xi));


    return prob;

} // compute_unemp_prob



double retirement_income(par_struct* par, double le){
    // compute retirement income

    double income = 0.0;
    double ratio = le / par->AE;
    if (ratio < 0.23){

        income = 0.9 * le ;
    }
    else if (ratio < 1.38){
        income = 0.2 * le + 0.32*(le - 0.23 * par->AE);
    }
    else {
        income = 0.57 * par->AE + 0.15 * (le - 1.38 * par->AE);

    }

    return income;



    
}

double compute_q(par_struct* par, egm_struct* egm, long long t,double m_pd, double z1, long long i_types ,double le){
    // compute q for a given state

    double q = 0.0; // initialize

    double alpha = par->alpha_types[i_types];
    double theta = par->theta_types[i_types];

    // a. working agents
    if (t < par->T_retired){
    for(long long i_unemployed = 0; i_unemployed < par->Nunemp+1; i_unemployed++){
    for(long long i_eta = 0; i_eta < par->Neta; i_eta++){
    for(long long i_epsilon = 0; i_epsilon < par->Nepsilon; i_epsilon++){
    for(long long i_mix_epsilon = 0; i_mix_epsilon < 2; i_mix_epsilon++){
    for(long long i_mix_eta = 0; i_mix_eta < 2; i_mix_eta++){

        // i. unpack
        double epsilon_base = par->epsilon[i_epsilon];
        double eta_base = par->eta[i_eta];
        double unemp = par->unemp[i_unemployed];

        // ii. transform epsilon nodes
        double epsilon = epsilon_base*par->sigma_epsilon_vec[i_mix_epsilon] + par->mu_epsilon_vec[i_mix_epsilon];
        
        // iii. transform eta nodes
        double eta = eta_base*par->sigma_eta_vec[i_mix_eta] + par->mu_eta_vec[i_mix_eta];

        // iv. next-period persistent income
        double z1_plus = par->rho_z * z1 + eta;
        if (z1_plus > egm->z1_max){
            z1_plus = egm->z1_max;
        }
        else if (z1_plus < egm->z1_min){
            z1_plus = egm->z1_min;
        }

        // v. income
        double income = (1-unemp) * exp(par->g[t+1] + alpha + theta*(t+1+1)/10 + z1_plus + epsilon);
        double disp_income = par->lambdaa * pow(MAX(income,par->Y_min), 1.0-par->tau);

        double le_plus = 0.0 ;
        if (par->use_reg == 1){
            le_plus = le;
        }
        else{
            le_plus = le / (t+2) * (t+1) + income / (t+2);

        }
        if (le_plus > egm->le_very_high){
            le_plus = egm->le_very_high;
        }
        else if (le_plus < egm->le_low){
            le_plus = egm->le_low;
        }
        
        // vi. next period cash-on-hand
        double m_plus = par->R*m_pd + disp_income;
        if (m_plus > egm->m_max_high){
            m_plus = egm->m_max_high;
        }



        // vii. next-period consumption from interpolation
        // get m-grid

        long long m_grid_index = index::d2(t+1,0,par->T,egm->Nm);
        
        long long i_sol_interp ;

        double c_plus = 0.0;
        if (par->use_reg == 0){
        i_sol_interp = index::d5(t+1,i_types,0,0,0,par->T,par->Ntypes,egm->Nle,egm->Nz,egm->Nm);
        c_plus = linear_interp::interp_3d(
            egm->le_grid,egm->z_grid, &egm->m_grids[m_grid_index], // grids
            egm->Nle, egm->Nz, egm->Nm, // dimensions
            &egm->sol_con[i_sol_interp], // sol_con
            le_plus, z1_plus,m_plus); // points
        }
        else{
             i_sol_interp = index::d4(t+1,i_types,0,0,par->T,par->Ntypes,egm->Nz,egm->Nm);
            c_plus = linear_interp::interp_2d(
            egm->z_grid, &egm->m_grids[m_grid_index], // grids
            egm->Nz, egm->Nm, // dimensions
            &egm->sol_con[i_sol_interp], // sol_con
            z1_plus,m_plus); // points
        }

        
        // viii. next-period marginal utility of consumption
        double mu_plus = marg_util_con(par,c_plus);

        // ix. compute probability of unemployment
        double prob_unemp = compute_unemp_prob(par,t,z1_plus);
        double unemp_w = unemp * prob_unemp + (1-unemp) * (1-prob_unemp);


        // ix. sum up
        q+=  par->R * par->s_p[t] *  unemp_w * par->eta_w[i_eta] * par->epsilon_w[i_epsilon] * par->p_epsilon_vec[i_mix_epsilon] * par->p_z_vec[i_mix_eta] * mu_plus;

    
    } // loop over mixtures for persistent shocks
    } // loop over mixtures for  for transitory shocks
    } // loop over transitory shocks
    } // loop over persistent shocks
    } // loop over unemployment shock

        // x. handle bequest if death
        double marg_bequest = 0.0;
        if (par->bequest_1 > 0.0){
            double bequest_taxed ; 
            if (m_pd > 0.0 ){

                bequest_taxed = m_pd * (1-par->tau_bequest);
                marg_bequest = par->bequest_1 *(1-par->tau_bequest) * pow(bequest_taxed + par->bequest_2,- par->gamma) ;
            }
            else{
                bequest_taxed = m_pd;
                marg_bequest = par->bequest_1 * pow(bequest_taxed + par->bequest_2,- par->gamma) ;
            }
            }

        q += (1.0-par->s_p[t]) * marg_bequest; // add weighted bequest utility

    } // t < T_retired condition

    // b. retired agents
    else {
        
        // i. income

        double le_use = le;
        if (par->use_reg == 1){

            long long constant_type_index = index::d2(i_types,0,par->Ntypes,2);
            long long coef_ficient_index = index::d2(i_types,1,par->Ntypes,2);
            double reg_constant = par->reg_coefs[constant_type_index];
            double reg_linear = par->reg_coefs[coef_ficient_index];
            le_use = reg_constant + reg_linear * z1;
            if (le_use < 0.0){
                le_use = 0.0;
            } // if le_use is negative

        }
        else{
            le_use = le;
        } // use le_use for retirement income
        double income = retirement_income(par,le_use);

        income = par->lambdaa * pow(MAX(income,par->Y_min), 1.0-par->tau);

        // ii. next period persistent income
        double z1_plus = z1;
        double le_plus = le;

        // iii. next period cash-on-hand
        double m_plus = par->R*m_pd + income;

        if (m_plus > egm->m_max_high){
            m_plus = egm->m_max_high;
        }
        
        // iv. next-period consumption from interpolation
        long long i_sol_interp ;
        i_sol_interp = index::d5(t+1,i_types,0,0,0,par->T,par->Ntypes,egm->Nle,egm->Nz,egm->Nm);

        long long m_grid_index = index::d2(t+1,0,par->T,egm->Nm);

        double c_plus = 0.0;
        if(par->use_reg == 0){

            c_plus = linear_interp::interp_3d(
            egm->le_grid, egm->z_grid, &egm->m_grids[m_grid_index], // grids
            egm->Nle,egm->Nz, egm->Nm, // dimensions
            &egm->sol_con[i_sol_interp], // sol_con
            le_plus, z1_plus,m_plus); // points
        }
        else{
            c_plus = linear_interp::interp_2d(
            egm->z_grid, &egm->m_grids[m_grid_index], // grids
            egm->Nz, egm->Nm, // dimensions
            &egm->sol_con[i_sol_interp], // sol_con
            z1_plus,m_plus); // points

        }

        // v. next-period marginal utility of consumption
        double mu_plus = marg_util_con(par,c_plus);

        // vi. sum up
        q += par->s_p[t] * par->R * mu_plus;


        double marg_bequest = 0.0;
        if (par->bequest_1 > 0){
            // marg_bequest = par->bequest_1 * pow(m_pd + par->bequest_2, -par->gamma) ;

            if (m_pd > 0.0 ){

                marg_bequest = (1-par->tau_bequest) * par->bequest_1 * pow(m_pd * (1-par->tau_bequest) + par->bequest_2, -par->gamma) ;
            }
            else{
                marg_bequest = par->bequest_1 * pow(m_pd + par->bequest_2, -par->gamma) ;
            }
        }
        q += (1.0-par->s_p[t]) * marg_bequest;
        
        } // retired

    return q;

} // compute q

void interp_to_common_grid( par_struct* par, egm_struct* egm, double* q_grid, 
                            double* m_temp, double* c_temp, long long q_index, long long sol_con_index,long long t){

    // a. unpack

    double* sol_con = egm->sol_con;

    // b. endogenous grid
    for (long long i_m_pd = 0; i_m_pd < egm->Nm_pd; i_m_pd++){

        // i. a_pd
        long long m_pd_index = index::d2(t,i_m_pd,par->T,egm->Nm_pd);
        double m_pd = egm->m_pd_grids[m_pd_index];
        long long i_q = q_index + i_m_pd;

        // ii. c
        // c_temp[i_m_pd+1] = inverse_marg_util_con(par,par->beta*par->R*q_grid[i_q]);
        c_temp[i_m_pd+1] = inverse_marg_util_con(par,par->beta* q_grid[i_q]);

        // iii. m
        m_temp[i_m_pd+1] = m_pd + c_temp[i_m_pd+1];

    } // m_pd loop

    // c. conversion to common grid
    long long m_grid_index = index::d2(t,0,par->T,egm->Nm);
    linear_interp::interp_1d_vec_mon_noprep(
        m_temp,egm->Nm_pd+1,c_temp,
        &egm->m_grids[m_grid_index],&sol_con[sol_con_index],egm->Nm
    );
    
    } 

EXPORT void solve_all(par_struct* par, egm_struct* egm){

    logs::write("log.txt",0,"par->cpp = %d threads\n",par->cppthreads);
    #pragma omp parallel num_threads(par->cppthreads)
    {

    double* m_temp = new double[egm->Nm_pd+1];
    double* c_temp = new double[egm->Nm_pd+1];
    
    // m_temp[0] = 0.0;
    c_temp[0] = 0.0;

    for(long long t = par->T-1; t >= 0; t--){

        #pragma omp master
        logs::write("log.txt",1,"t = %d\n",t);
        m_temp[0] = par->a_low[t];

        // a. last period
        if(t == par->T-1){
            
            #pragma omp for collapse(4)
            for(long long i_z = 0; i_z < egm->Nz; i_z++){
            for(long long i_types = 0; i_types < par->Ntypes; i_types++){
            for(long long i_m = 0; i_m < egm->Nm; i_m++){
            for (long long i_le = 0; i_le < egm->Nle; i_le++){

                // consume everything
                long long i_sol;
                i_sol = index::d5(
                    t,i_types,i_le,i_z,i_m,
                    par->T,par->Ntypes,egm->Nle,egm->Nz,egm->Nm
                );

                long long m_grid_index = index::d2(
                    t,i_m,par->T,egm->Nm
                );
                double m = egm->m_grids[m_grid_index];

                double target = 0.0;
                if (par->bequest_1 == 0){

                    target = m;
                }
                else{
                    target = 1 / (pow((1-par->tau_bequest) * par->bequest_1*par->beta,1/par->gamma) + (1-par->tau_bequest)) * (m*(1-par->tau_bequest) + par->bequest_2);
                    if (target > m){
                        target = m;
                    }
                }

                egm->sol_con[i_sol] = target;

            } // m-loop
            } // type loop
            } // z_loop
            } // le loop

        } // final period

        // b. other periods
        else {


            // i. Initialize q_grid
            long long q_size = par->Ntypes * egm->Nle * egm->Nz * egm->Nm_pd;
            double* q_grid = new double[q_size];

            // ii. compute q
            #pragma omp for collapse(4)
            for (long long i_z = 0; i_z < egm->Nz; i_z++){
            for (long long i_types = 0; i_types < par->Ntypes; i_types++){
            for (long long i_le = 0; i_le < egm->Nle; i_le++){
            for (long long i_m_pd = 0; i_m_pd < egm->Nm_pd; i_m_pd++){
            
                double z = egm->z_grid[i_z];
                double le = egm->le_grid[i_le];
                // long long m_pd_index = (t) * egm->Nm_pd;
                long long m_pd_index = index::d2(
                    t,i_m_pd,
                    par->T,egm->Nm_pd
                );
                double m_pd = egm->m_pd_grids[m_pd_index];

                double q = compute_q(par,egm,t,m_pd,z,i_types,le);

                long long i_q;
                i_q = index::d4(i_types,i_le, i_z,i_m_pd,
                    par->Ntypes,egm->Nle, egm->Nz,egm->Nm_pd
                );                
                q_grid[i_q] = q;
                                        

            } // m_pd loop
            } // type loop
            } // z loop
            } // le loop
            
            // iii. endogenous grid and conversion to common grid
            #pragma omp for collapse(3)
            for (long long i_z = 0; i_z < egm->Nz; i_z++){
            for (long long i_types = 0; i_types < par->Ntypes; i_types++){
            for (long long i_le = 0; i_le < egm->Nle; i_le++){
            
                long long q_index = index::d4(
                    i_types,i_le, i_z,0,
                    par->Ntypes,egm->Nle,egm->Nz,egm->Nm_pd
                );
                
            long long sol_con_index;
            sol_con_index = index::d5(
                t,i_types,i_le,i_z,0,
                par->T,par->Ntypes,egm->Nle,egm->Nz,egm->Nm
            );
                
                interp_to_common_grid(par,egm,q_grid,m_temp,c_temp,q_index,sol_con_index,t);
            

            } // type loop
            } // z1 loop
            } // le loop
            
            delete[] q_grid;

        } // not last period
        
    } // t

    delete[] m_temp;
    delete[] c_temp;


    } // parallel condition

} // solve





EXPORT void compute_euler_errors(par_struct* par, egm_struct* egm, sim_struct* sim) {
    logs::write("log.txt", 0, "Computing Euler errors from simulated data with %d threads\n", par->cppthreads);
    
    #pragma omp parallel num_threads(par->cppthreads)
    {
        // Loop over all time periods except last (no Euler equation in last period)
        #pragma omp for collapse(1)
        for (long long t = 0; t < par->T - 1; t++) {            
            for (long long i = 0; i < sim->N; i++) {
                // Current state and consumption
                long long state_index = index::d3(t, i, 0, par->T, sim->N, par->Nstates);
                long long outcome_index = index::d3(t, i, 0, par->T, sim->N, par->Noutcomes);
                
                double c_ti = sim->outcomes[outcome_index];
                double m_ti = sim->states[state_index + 0];  // m is first state variable
                double a_ti = m_ti - c_ti;  // savings = m - c
                double z_ti = sim->states[state_index + 1];  // persistent income shock
                double le_ti = sim->states[state_index + 2]; // labor efficiency
                
                // Find active type (one-hot encoded in remaining state variables)
                long long type_ti = -1;
                for (long long j = 0; j < par->Ntypes; j++) {
                    if (sim->states[state_index + 3 + j] > 0.5) { // Check if dummy is active
                        type_ti = j;
                        break;
                    }
                }
                
                // Default to type 0 if none found (shouldn't happen with proper one-hot encoding)
                if (type_ti == -1) {
                    type_ti = 0;
                }
                
                // Reuse the original compute_q function
                double q = compute_q(par, egm, t, a_ti, z_ti, type_ti, le_ti);
                
                // Compute Euler error: (c_t+1^*/c_t) - 1, where c_t+1^* is optimal consumption implied by Euler equation
                double mu_current = marg_util_con(par, c_ti);
                double c_star = inverse_marg_util_con(par, par->beta * q);
                double error = (c_star / c_ti) - 1.0;
                
                // Store error (log10 of absolute error)
                long long error_index = index::d2(t, i, par->T, sim->N);
                sim->euler_error[error_index] = fabs(error) ;
            }
        }
    }
}