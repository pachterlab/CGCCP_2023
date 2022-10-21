pip install scanpy -q
pip install scvi-tools==0.8.1 -q
pip install loompy -q
pip install leidenalg -q
#pip install --upgrade torch==1.12.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html -q

python3 Analysis2_Train.py --name 'const_5ct'

python3 Analysis2_Train.py --name 'BVNB_5ct'

python3 Analysis2_Train.py --name 'bursty_5ct'


python3 Analysis2_Train.py --name B08_processed_hv --data_dir ../data/allen/


python3 Analysis2_Train.py --name 'const_20ct'

python3 Analysis2_Train.py --name 'BVNB_20ct'

python3 Analysis2_Train.py --name 'bursty_20ct'


python3 Analysis2_Train.py --name 'const_10ct'

python3 Analysis2_Train.py --name 'BVNB_10ct'

python3 Analysis2_Train.py --name 'bursty_10ct'


python3 Analysis2_Train.py --name 'const_15ct'

python3 Analysis2_Train.py --name 'BVNB_15ct'

python3 Analysis2_Train.py --name 'bursty_15ct'


python3 Analysis2_Train.py --name A08_processed_hv --data_dir ../data/allen/
python3 Analysis2_Train.py --name B01_processed_hv --data_dir ../data/allen/
python3 Analysis2_Train.py --name C01_processed_hv --data_dir ../data/allen/
python3 Analysis2_Train.py --name F08_processed_hv --data_dir ../data/allen/




