Folder containing scripts and notebooks for data simulation, preprocessing, and training biVI models.


Simulated data generation:
`Define_Simulation_Parameters`: analysis of experimental data sets to determine reasonable parameters for simulation
`Simulate_data.ipynb`: notebook to simulate bursty, constitutive, and extrinsic parameters and count matrices

Allen data:
`Preprocess_Data.ipynb`: notebeook identical to the script `Preprocess_Data.py` called by `preprocess.sh` to process Allen 10x datasets

Data availability:
Simulated data and processed Allen data (sample B08) can be found in the Zenodo package 7497222. 

Training Models:
`Train.ipynb`: training notebook to explore various configurations and troubleshoot
`train_biVI.py`: training script run 
`train.sh`: bash script


Analyses:
`kld.py` and `kld.sh`: scripts to calculate kld between simulated and ground truth distributions for simulated data 
`Analysis1a_Simulated_Bursty.ipynb`: calculates metrics and generates plots for supplementary figure for bursty simulated data
`Analysis1b_Simulated_Extrinsic.ipynb`: calculates metrics and generates plots for supplementary figure for constitutive simulated data
`Analysis1c_Simulated_Constitutive.ipynb`: calculates metrics and generates plots for supplementary figure for extrinsic simulated data
`Analysis2_Allen.ipynb`: calculates metrics and generates plots for supplementary figure for Allen data sample B08
`Analysis3_Simulated_Distributions.ipynb`: visualization of reconstructed and sampled distributions, spread of inferred parameters for bursty simulated data, preprint Figure 2
`Analysis4_Allen_Distributions.ipynb`: visualization of reconstructed and sampled distributions, spread of inferred parameters for Allen data sample B08, preprint Figure 3
`Analysis5_DifferentialExpression.ipynb`: calculates differential expression of inferred parameters for Allen data sample B08












