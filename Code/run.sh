## 10x datasets
datasets=(pbmc_10k_v3 pbmc_1k_v3 \
          heart_10k_v3 heart_1k_v3 \
          neuron_10k_v3 neuron_1k_v3 \
          brain_5k brain_nuc_5k.loom)

datasets=( "pbmc_10k_v3" )

for dataset in "${datasets[@]}"
do
  logdir="out/${dataset}/data"
  loomfile="data/loom_10x_kb/${dataset}.loom"

  python preprocess.py --logdir "${logdir}" \
                       --loomfile "${loomfile}"

  python run_scBIVI.py --datadir "${logdir}/preprocessed.h5ad" \
                       --percent_keep "1" \
                       --cluster_method 'RNA_leiden'
done

## For guide RNA based clustering
logdir='out/scbivi_gRNA/data'
loomfile='data/scbivi_gRNA.loom'

python preprocess.py --logdir "${logdir}" \
                     --loomfile "${loomfile}"

python run_scBIVI.py --datadir "${logdir}/preprocessed.h5ad" \
                     --percent_keep "1" \
                     --cluster_method 'Guide'
