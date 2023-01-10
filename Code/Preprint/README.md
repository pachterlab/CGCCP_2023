Folder containing scripts and notebooks for data simulation, preprocessing, and training biVI models.


Simulated data generation:
* `Define_Simulation_Parameters.ipynb`: analysis of experimental data sets to determine reasonable parameters for simulation <br />
* `Simulate_data.ipynb`: notebook to simulate bursty, constitutive, and extrinsic parameters and count matrices <br />

Allen data:
* `Preprocess_Data.ipynb`: notebeook identical to the script `Preprocess_Data.py` called by `preprocess.sh` to process Allen 10x datasets

Data availability:
* Simulated data and processed Allen data (sample B08) can be found in the Zenodo package [7497222](https://zenodo.org/record/7497222)

Training Models:
* `Train.ipynb`: training notebook to explore various configurations and troubleshoot <br />
* `train_biVI.py`: training script run <br />
* `train.sh`: bash script <br />


Analyses:
* `kld.py` and `kld.sh`: scripts to calculate kld between simulated and ground truth distributions for simulated data <br />
* `Analysis1a_Simulated_Bursty.ipynb`: calculates metrics and generates plots for supplementary figure for bursty simulated data <br />
* `Analysis1b_Simulated_Extrinsic.ipynb`: calculates metrics and generates plots for supplementary figure for constitutive simulated data <br />
* `Analysis1c_Simulated_Constitutive.ipynb`: calculates metrics and generates plots for supplementary figure for extrinsic simulated data <br />
* `Analysis2_Allen.ipynb`: calculates metrics and generates plots for supplementary figure for Allen data sample B08 <br />
* `Analysis3_Simulated_Distributions.ipynb`: visualization of reconstructed and sampled distributions, spread of inferred parameters for bursty simulated data, preprint Figure 2 <br />
* `Analysis4_Allen_Distributions.ipynb`: visualization of reconstructed and sampled distributions, spread of inferred parameters for Allen data sample B08, preprint Figure 3 <br />
* `Analysis5_DifferentialExpression.ipynb`: calculates differential expression of inferred parameters for Allen data sample B08 <br />












