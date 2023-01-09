# names=("bursty_20ct_many" "const_20ct_many" "extrinsic_20ct_many") 
names=("const_20ct_many" "extrinsic_20ct_many") 

for name in "${names[@]}"

do
  echo ${name}
  python3 kld.py --data_dir "../../data/simulated_data/" --name ${name}

done
                       