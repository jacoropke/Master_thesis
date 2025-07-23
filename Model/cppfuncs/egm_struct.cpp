typedef struct egm_struct
{
 int Nm_low;
 int Nm_high;
 int Nm;
 int Nm_pd_low;
 int Nm_pd_high;
 int Nm_pd;
 int Nz;
 double m_max_normal;
 double m_max_high;
 double le_low;
 double le_high;
 double le_very_high;
 int Nle;
 double z1_max;
 double z1_min;
 double* m_pd_grids;
 double* m_grids;
 double* z_grid;
 double* le_grid;
 double* sol_con;
 double* transfer_grid;
 int Ntransfer;
} egm_struct;

int get_int_egm_struct(egm_struct* x, char* name){

 if( strcmp(name,"Nm_low") == 0 ){ return x->Nm_low; }
 else if( strcmp(name,"Nm_high") == 0 ){ return x->Nm_high; }
 else if( strcmp(name,"Nm") == 0 ){ return x->Nm; }
 else if( strcmp(name,"Nm_pd_low") == 0 ){ return x->Nm_pd_low; }
 else if( strcmp(name,"Nm_pd_high") == 0 ){ return x->Nm_pd_high; }
 else if( strcmp(name,"Nm_pd") == 0 ){ return x->Nm_pd; }
 else if( strcmp(name,"Nz") == 0 ){ return x->Nz; }
 else if( strcmp(name,"Nle") == 0 ){ return x->Nle; }
 else if( strcmp(name,"Ntransfer") == 0 ){ return x->Ntransfer; }
 else {return -9999;}

}


double get_double_egm_struct(egm_struct* x, char* name){

 if( strcmp(name,"m_max_normal") == 0 ){ return x->m_max_normal; }
 else if( strcmp(name,"m_max_high") == 0 ){ return x->m_max_high; }
 else if( strcmp(name,"le_low") == 0 ){ return x->le_low; }
 else if( strcmp(name,"le_high") == 0 ){ return x->le_high; }
 else if( strcmp(name,"le_very_high") == 0 ){ return x->le_very_high; }
 else if( strcmp(name,"z1_max") == 0 ){ return x->z1_max; }
 else if( strcmp(name,"z1_min") == 0 ){ return x->z1_min; }
 else {return NAN;}

}


double* get_double_p_egm_struct(egm_struct* x, char* name){

 if( strcmp(name,"m_pd_grids") == 0 ){ return x->m_pd_grids; }
 else if( strcmp(name,"m_grids") == 0 ){ return x->m_grids; }
 else if( strcmp(name,"z_grid") == 0 ){ return x->z_grid; }
 else if( strcmp(name,"le_grid") == 0 ){ return x->le_grid; }
 else if( strcmp(name,"sol_con") == 0 ){ return x->sol_con; }
 else if( strcmp(name,"transfer_grid") == 0 ){ return x->transfer_grid; }
 else {return NULL;}

}


