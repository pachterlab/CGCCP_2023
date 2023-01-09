# pip install scanpy -q
# pip install scvi-tools==0.8.1 -q
# pip install loompy -q
# pip install leidenalg -q
#pip install --upgrade torch==1.12.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html -q


# simulated data 
python3 train_biVI.py --name 'const_20ct_many' --data_dir ../../data/simulated_data/
python3 train_biVI.py --name 'extrinsic_20ct_many' --data_dir ../../data/simulated_data/
python3 train_biVI.py --name 'bursty_20ct_many' --data_dir ../../data/simulated_data/


# allen data
python3 train_biVI.py --name B08_processed_hv --data_dir ../../data/allen/
# python3 train_biVI.py --name A08_processed_hv --data_dir ../../data/allen/
# python3 train_biVI.py --name B01_processed_hv --data_dir ../../data/allen/
# python3 train_biVI.py --name C01_processed_hv --data_dir ../../data/allen/
# python3 train_biVI.py --name F08_processed_hv --data_dir ../../data/allen/
# python3 train_biVI.py --name H12_processed_hv --data_dir ../../data/allen/


# store figures for each of the 

