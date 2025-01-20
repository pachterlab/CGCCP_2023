from setuptools import setup,find_packages

long_description = "Variational autoencoder for analyzing nascent and mature single cell RNA sequencing data using biophysical models."

setup(
    name="BIVI",  # Required
    version="0.1.0",  # Required
    description="biVI software",  # Optional
    url="https://github.com/pachterlab/CGCCP_2023/tree/main/BIVI",
    author="Carilli, Maria, and Gorin, Gennady, and Choi, Yongin, and Chari, Tara, and Pachter, Lior",
    packages=find_packages(),  # Required
#     py_modules=['bivae','biVI','distributiosns','nnNB_module'],
    # package_dir={'BIVI': './BIVI'},
    python_requires=">=3.7, <4",
    install_requires=['scanpy',
                      'numpy',
                      'pandas',
                      'scvi-tools'
                      # 'scvi-tools==0.18.0',
                      # 'torchmetrics>=0.7.0,<0.9.0',
                      'loompy'],  # Optional
    package_data={"":['*zip']}, # Optional, required if package needs non python script files
    # include_package_data=True,
    # data_files = [('model', ['./BIVI/models/best_model_MODEL.zip'])],
)
