# CGCCP_2023
This repository contains the scripts and notebooks for the preprint "Biophysical modeling with variational autoencoders for bimodal, single-cell RNA sequencing data". Although variational autoencoders can be treated as purely phenomenological and descriptive, without any explicit claims about the processes that generated data, it is possible to exploit *implicit* physics encoded in the mathematical formulation to model biophysical processes that gave rise to observations. By interpreting the _scVI_ generative model as a description of a particular biophysical model, we can represent bivariate RNA distributions. We benchmark the implementation, _biVI_, on simulated and biological data.

`BIVI/` contains all of the scripts used to implement _biVI_, while `Manuscript/analysis/` contains all of the notebooks and scripts used to generate the manuscript figures and results. `Example/` contains `kb_pipeline.sh,` a script that demonstrates how to align raw reads to a reference genome to obtain the unspliced/spliced count matrices necessary for _biVI_, and 'Demo.ipynb,` a Google Colaboratory notebook that processes the output matrices, train a _biVI_ model, and visualize the results.  





The biVI software can be installed as a standalone package using the following command: 


<code> pip3 install git+https://github.com/pachterlab/CGCCP_2023.git#subdirectory=BIVI </code>. 



If package dependencies cause installation issues, create a clean Conda environment and rerun the installation:


<code> conda create --name biVI_env python==3.9 </code> 

<code> pip3 install git+https://github.com/pachterlab/CGCCP_2023.git#subdirectory=BIVI </code>.


Installation takes one to several minutes on a standard laptop. Alternatively, _biVI_ can be run in a Google Colab notebook, an example of which is given in `Example/Demo.ipynb`.
