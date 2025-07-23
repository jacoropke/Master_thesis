typedef struct par_struct
{
 bool full;
 int seed;
 int T;
 int T_retired;
 double beta;
 double bequest_1;
 double bequest_2;
 double gamma;
 double omega_0;
 double omega_1;
 double omega_2;
 double tau;
 double lambdaa;
 double Y_min;
 double le_scaler;
 double tau_bequest;
 double R;
 double borr_tightness;
 double p_epsilon;
 double mu_epsilon_1;
 double sigma_epsilon_1;
 double sigma_epsilon_2;
 int Nepsilon;
 double rho_z;
 double mu_eta_1;
 double p_z;
 double sigma_eta_1;
 double sigma_eta_2;
 int Neta;
 bool disable_z_income;
 int Nunemp;
 double a_p_unemp;
 double b_p_unemp;
 double c_p_unemp;
 double d_p_unemp;
 int Nalpha_types;
 int Ntheta_types;
 double sigma_alpha;
 double sigma_theta;
 double corr_alpha_theta;
 bool set_type_probs;
 double* type_probs_assign;
 double mu_z1_0;
 double sigma_z1_0;
 double AE;
 bool use_reg;
 int Nstates;
 int Nstates_pd;
 int Nshocks;
 int Noutcomes;
 bool KKT;
 int NDC;
 double m_scaler;
 double p_scaler;
 char* policy_predict;
 bool death;
 int cppthreads;
 double* g;
 double mu_eta_2;
 double mu_epsilon_2;
 int Nactions;
 double* epsilon;
 double* epsilon_w;
 double* eta;
 double* eta_w;
 double* unemp;
 double* unemp_w;
 double* a_low;
 double* p_z_vec;
 double* mix_eta;
 double* p_epsilon_vec;
 double* mix_epsilon;
 double* sigma_eta_vec;
 double* sigma_epsilon_vec;
 double* mu_eta_vec;
 double* mu_epsilon_vec;
 int Ntypes;
 double* alpha_types;
 double* theta_types;
 double* type_probs;
 double* Y_min_tensor;
 double* scale_vec_states;
 double* scale_vec_states_pd;
 double* survival_probs;
 double* s_p;
 double* s_p_torch;
 double* uncon_survival;
 double* reg_coefs;
} par_struct;

bool get_bool_par_struct(par_struct* x, char* name){

 if( strcmp(name,"full") == 0 ){ return x->full; }
 else if( strcmp(name,"disable_z_income") == 0 ){ return x->disable_z_income; }
 else if( strcmp(name,"set_type_probs") == 0 ){ return x->set_type_probs; }
 else if( strcmp(name,"use_reg") == 0 ){ return x->use_reg; }
 else if( strcmp(name,"KKT") == 0 ){ return x->KKT; }
 else if( strcmp(name,"death") == 0 ){ return x->death; }
 else {return false;}

}


int get_int_par_struct(par_struct* x, char* name){

 if( strcmp(name,"seed") == 0 ){ return x->seed; }
 else if( strcmp(name,"T") == 0 ){ return x->T; }
 else if( strcmp(name,"T_retired") == 0 ){ return x->T_retired; }
 else if( strcmp(name,"Nepsilon") == 0 ){ return x->Nepsilon; }
 else if( strcmp(name,"Neta") == 0 ){ return x->Neta; }
 else if( strcmp(name,"Nunemp") == 0 ){ return x->Nunemp; }
 else if( strcmp(name,"Nalpha_types") == 0 ){ return x->Nalpha_types; }
 else if( strcmp(name,"Ntheta_types") == 0 ){ return x->Ntheta_types; }
 else if( strcmp(name,"Nstates") == 0 ){ return x->Nstates; }
 else if( strcmp(name,"Nstates_pd") == 0 ){ return x->Nstates_pd; }
 else if( strcmp(name,"Nshocks") == 0 ){ return x->Nshocks; }
 else if( strcmp(name,"Noutcomes") == 0 ){ return x->Noutcomes; }
 else if( strcmp(name,"NDC") == 0 ){ return x->NDC; }
 else if( strcmp(name,"cppthreads") == 0 ){ return x->cppthreads; }
 else if( strcmp(name,"Nactions") == 0 ){ return x->Nactions; }
 else if( strcmp(name,"Ntypes") == 0 ){ return x->Ntypes; }
 else {return -9999;}

}


double get_double_par_struct(par_struct* x, char* name){

 if( strcmp(name,"beta") == 0 ){ return x->beta; }
 else if( strcmp(name,"bequest_1") == 0 ){ return x->bequest_1; }
 else if( strcmp(name,"bequest_2") == 0 ){ return x->bequest_2; }
 else if( strcmp(name,"gamma") == 0 ){ return x->gamma; }
 else if( strcmp(name,"omega_0") == 0 ){ return x->omega_0; }
 else if( strcmp(name,"omega_1") == 0 ){ return x->omega_1; }
 else if( strcmp(name,"omega_2") == 0 ){ return x->omega_2; }
 else if( strcmp(name,"tau") == 0 ){ return x->tau; }
 else if( strcmp(name,"lambdaa") == 0 ){ return x->lambdaa; }
 else if( strcmp(name,"Y_min") == 0 ){ return x->Y_min; }
 else if( strcmp(name,"le_scaler") == 0 ){ return x->le_scaler; }
 else if( strcmp(name,"tau_bequest") == 0 ){ return x->tau_bequest; }
 else if( strcmp(name,"R") == 0 ){ return x->R; }
 else if( strcmp(name,"borr_tightness") == 0 ){ return x->borr_tightness; }
 else if( strcmp(name,"p_epsilon") == 0 ){ return x->p_epsilon; }
 else if( strcmp(name,"mu_epsilon_1") == 0 ){ return x->mu_epsilon_1; }
 else if( strcmp(name,"sigma_epsilon_1") == 0 ){ return x->sigma_epsilon_1; }
 else if( strcmp(name,"sigma_epsilon_2") == 0 ){ return x->sigma_epsilon_2; }
 else if( strcmp(name,"rho_z") == 0 ){ return x->rho_z; }
 else if( strcmp(name,"mu_eta_1") == 0 ){ return x->mu_eta_1; }
 else if( strcmp(name,"p_z") == 0 ){ return x->p_z; }
 else if( strcmp(name,"sigma_eta_1") == 0 ){ return x->sigma_eta_1; }
 else if( strcmp(name,"sigma_eta_2") == 0 ){ return x->sigma_eta_2; }
 else if( strcmp(name,"a_p_unemp") == 0 ){ return x->a_p_unemp; }
 else if( strcmp(name,"b_p_unemp") == 0 ){ return x->b_p_unemp; }
 else if( strcmp(name,"c_p_unemp") == 0 ){ return x->c_p_unemp; }
 else if( strcmp(name,"d_p_unemp") == 0 ){ return x->d_p_unemp; }
 else if( strcmp(name,"sigma_alpha") == 0 ){ return x->sigma_alpha; }
 else if( strcmp(name,"sigma_theta") == 0 ){ return x->sigma_theta; }
 else if( strcmp(name,"corr_alpha_theta") == 0 ){ return x->corr_alpha_theta; }
 else if( strcmp(name,"mu_z1_0") == 0 ){ return x->mu_z1_0; }
 else if( strcmp(name,"sigma_z1_0") == 0 ){ return x->sigma_z1_0; }
 else if( strcmp(name,"AE") == 0 ){ return x->AE; }
 else if( strcmp(name,"m_scaler") == 0 ){ return x->m_scaler; }
 else if( strcmp(name,"p_scaler") == 0 ){ return x->p_scaler; }
 else if( strcmp(name,"mu_eta_2") == 0 ){ return x->mu_eta_2; }
 else if( strcmp(name,"mu_epsilon_2") == 0 ){ return x->mu_epsilon_2; }
 else {return NAN;}

}


double* get_double_p_par_struct(par_struct* x, char* name){

 if( strcmp(name,"type_probs_assign") == 0 ){ return x->type_probs_assign; }
 else if( strcmp(name,"g") == 0 ){ return x->g; }
 else if( strcmp(name,"epsilon") == 0 ){ return x->epsilon; }
 else if( strcmp(name,"epsilon_w") == 0 ){ return x->epsilon_w; }
 else if( strcmp(name,"eta") == 0 ){ return x->eta; }
 else if( strcmp(name,"eta_w") == 0 ){ return x->eta_w; }
 else if( strcmp(name,"unemp") == 0 ){ return x->unemp; }
 else if( strcmp(name,"unemp_w") == 0 ){ return x->unemp_w; }
 else if( strcmp(name,"a_low") == 0 ){ return x->a_low; }
 else if( strcmp(name,"p_z_vec") == 0 ){ return x->p_z_vec; }
 else if( strcmp(name,"mix_eta") == 0 ){ return x->mix_eta; }
 else if( strcmp(name,"p_epsilon_vec") == 0 ){ return x->p_epsilon_vec; }
 else if( strcmp(name,"mix_epsilon") == 0 ){ return x->mix_epsilon; }
 else if( strcmp(name,"sigma_eta_vec") == 0 ){ return x->sigma_eta_vec; }
 else if( strcmp(name,"sigma_epsilon_vec") == 0 ){ return x->sigma_epsilon_vec; }
 else if( strcmp(name,"mu_eta_vec") == 0 ){ return x->mu_eta_vec; }
 else if( strcmp(name,"mu_epsilon_vec") == 0 ){ return x->mu_epsilon_vec; }
 else if( strcmp(name,"alpha_types") == 0 ){ return x->alpha_types; }
 else if( strcmp(name,"theta_types") == 0 ){ return x->theta_types; }
 else if( strcmp(name,"type_probs") == 0 ){ return x->type_probs; }
 else if( strcmp(name,"Y_min_tensor") == 0 ){ return x->Y_min_tensor; }
 else if( strcmp(name,"scale_vec_states") == 0 ){ return x->scale_vec_states; }
 else if( strcmp(name,"scale_vec_states_pd") == 0 ){ return x->scale_vec_states_pd; }
 else if( strcmp(name,"survival_probs") == 0 ){ return x->survival_probs; }
 else if( strcmp(name,"s_p") == 0 ){ return x->s_p; }
 else if( strcmp(name,"s_p_torch") == 0 ){ return x->s_p_torch; }
 else if( strcmp(name,"uncon_survival") == 0 ){ return x->uncon_survival; }
 else if( strcmp(name,"reg_coefs") == 0 ){ return x->reg_coefs; }
 else {return NULL;}

}


char* get_char_p_par_struct(par_struct* x, char* name){

 if( strcmp(name,"policy_predict") == 0 ){ return x->policy_predict; }
 else {return NULL;}

}


