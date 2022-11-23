names=("bursty_20ct_many" "const_20ct_many" "extrinsic_20ct_many") 

for name in "${names[@]}"

do
  echo ${name}
  python3 Analysis2_MSE.py --data_dir "../../data/simulated_data/" \
                       --name ${name}

  python3 Analysis1_ClusteringAccuracy.py --data_dir "../../data/simulated_data/" \
                       --name ${name} --index "test"
                       
  python3 Analysis1_ClusteringAccuracy.py --data_dir "../../data/simulated_data/" \
                       --name ${name} --index "train"


done

python3 Analysis2_MSE.py --data_dir "../../data/allen/" \
                       --name "B08_processed_hv"

python3 Analysis1_ClusteringAccuracy.py --data_dir "../../data/allen/" \
                       --name "B08_processed_hv" --index "test"
                      
python3 Analysis1_ClusteringAccuracy.py --data_dir "../../data/allen/" \
                       --name "B08_processed_hv" --index "train"
