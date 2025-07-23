# Master_thesis

The code is run through 4 notebooks:

01_Income_process.ipynb: This generates all results that are purely from the income process and do not require solving the model. This can be run without running the other notebooks.

02_Solve_DP.ipynb: This solves the models using DP and saves the results in pickle-files. 

03_Solve_DL.ipynb: This solves the models using DL and saves the results in pickle files. 

04_Results.ipynb: This generates all results not generated in 01_Income_process.ipynb. Typically results that require solving the model. Results are based upon pickle files generated in notebooks 02 and 03 so these must be run first. 


Running the DP-code:

DP-code is written in C++. I run this in python using the EconModel package in python. Running C++ requires a compiler. It is not designed to run on a GPU.

Running the DL-code:
The DL-code can in principle be run on a CPU but will be very slow. I recommend using some sort of GPU with cuda. The code is based on pytorch. The code is written as part of a package. The provided code has a folder called EconDLSolvers. The DeepSimEGM code is inside the package. installing it requires navigating to the folder in anaconda prompt (or similar alternative) and using the command "pip install -e .".