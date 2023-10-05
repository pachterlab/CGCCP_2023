names=("A08" "B08" "B01" "C01" "F08" "H12" "A02") 

for name in "${names[@]}"
do
#   logdir="out/${dataset}/data"
#   loomfile="data/loom_10x_kb/allen_${name}_raw.loom"

  python3 preprocess.py --data_dir "../../data/allen/" \
                       --name ${name}

#   python run_scBIVI.py --datadir "${logdir}/preprocessed.h5ad" \
#                        --percent_keep "1" \
#                        --cluster_method 'RNA_leiden'

done
