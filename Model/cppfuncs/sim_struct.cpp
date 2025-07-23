typedef struct sim_struct
{
 int N;
 int reps;
 double* states;
 double* states_pd;
 double* shocks;
 double* outcomes;
 double* actions;
 double* reward;
 double* income;
 double* unemployment_value_0;
 double* working_income_after_unemp;
 double* working_income_before_unemp;
 double* euler_error;
 double R;
 double* R_transfer;
 double* R_transfers;
 double* alpha;
 double* theta;
 double* eta;
 double* epsilon;
 double* deterministic_income;
 double* fixed_effect_income_type;
 double* trend_income_type;
 double* transitory_income;
 double* persistent_income;
 double* unemployment;
 double* lifetime_earnings;
 double* lifetime_earnings_compare;
 double* years_employed;
 double* terminal_reward;
} sim_struct;

int get_int_sim_struct(sim_struct* x, char* name){

 if( strcmp(name,"N") == 0 ){ return x->N; }
 else if( strcmp(name,"reps") == 0 ){ return x->reps; }
 else {return -9999;}

}


double* get_double_p_sim_struct(sim_struct* x, char* name){

 if( strcmp(name,"states") == 0 ){ return x->states; }
 else if( strcmp(name,"states_pd") == 0 ){ return x->states_pd; }
 else if( strcmp(name,"shocks") == 0 ){ return x->shocks; }
 else if( strcmp(name,"outcomes") == 0 ){ return x->outcomes; }
 else if( strcmp(name,"actions") == 0 ){ return x->actions; }
 else if( strcmp(name,"reward") == 0 ){ return x->reward; }
 else if( strcmp(name,"income") == 0 ){ return x->income; }
 else if( strcmp(name,"unemployment_value_0") == 0 ){ return x->unemployment_value_0; }
 else if( strcmp(name,"working_income_after_unemp") == 0 ){ return x->working_income_after_unemp; }
 else if( strcmp(name,"working_income_before_unemp") == 0 ){ return x->working_income_before_unemp; }
 else if( strcmp(name,"euler_error") == 0 ){ return x->euler_error; }
 else if( strcmp(name,"R_transfer") == 0 ){ return x->R_transfer; }
 else if( strcmp(name,"R_transfers") == 0 ){ return x->R_transfers; }
 else if( strcmp(name,"alpha") == 0 ){ return x->alpha; }
 else if( strcmp(name,"theta") == 0 ){ return x->theta; }
 else if( strcmp(name,"eta") == 0 ){ return x->eta; }
 else if( strcmp(name,"epsilon") == 0 ){ return x->epsilon; }
 else if( strcmp(name,"deterministic_income") == 0 ){ return x->deterministic_income; }
 else if( strcmp(name,"fixed_effect_income_type") == 0 ){ return x->fixed_effect_income_type; }
 else if( strcmp(name,"trend_income_type") == 0 ){ return x->trend_income_type; }
 else if( strcmp(name,"transitory_income") == 0 ){ return x->transitory_income; }
 else if( strcmp(name,"persistent_income") == 0 ){ return x->persistent_income; }
 else if( strcmp(name,"unemployment") == 0 ){ return x->unemployment; }
 else if( strcmp(name,"lifetime_earnings") == 0 ){ return x->lifetime_earnings; }
 else if( strcmp(name,"lifetime_earnings_compare") == 0 ){ return x->lifetime_earnings_compare; }
 else if( strcmp(name,"years_employed") == 0 ){ return x->years_employed; }
 else if( strcmp(name,"terminal_reward") == 0 ){ return x->terminal_reward; }
 else {return NULL;}

}


double get_double_sim_struct(sim_struct* x, char* name){

 if( strcmp(name,"R") == 0 ){ return x->R; }
 else {return NAN;}

}


