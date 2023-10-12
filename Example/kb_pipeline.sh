# bash script to produce unspliced/spliced count matrices with kb-python version 0.26.0

# install kb-python 
pip install kb-python==0.26.0


# define main path and number of threads
main_path='/home/tara/temp_git2/CGCCP_2023/Example'
threads=24




# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# creating the nascent/mature indices --> this only has to be done once for a reference genome

# download reference genome and annotations to $main_path/references/
mkdir -p $main_path/references
cd $main_path/references
wget https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_38/GRCh38.primary_assembly.genome.fa.gz
wget https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_38/gencode.v38.primary_assembly.annotation.gtf.gz


# create nascent and mature indices 
cd $main_path
mkdir -p $main_path/indices
kb ref --workflow=lamanno --verbose --overwrite -i $main_path/indices/human_lamanno.idx -g $main_path/indices/human_lamanno.t2g -c1 $main_path/indices/human_lamanno.mature.t2c -c2 $main_path/indices/human_lamanno.nascent.t2c -f1 $main_path/indices/human.lamanno.mature.fa -f2 $main_path/indices/human.lamanno.nascent.fa $main_path/references/GRCh38.primary_assembly.genome.fa.gz $main_path/references/gencode.v38.primary_assembly.annotation.gtf.gz



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# download 10x v3 1k PBMC raw data (change paths to reflect desired data)

mkdir -p $main_path/pbmc_1k_v3_raw
cd $main_path/pbmc_1k_v3_raw/
curl -O https://cf.10xgenomics.com/samples/cell-exp/3.0.0/pbmc_1k_v3/pbmc_1k_v3_fastqs.tar # 5-10 minutes
tar -xvf pbmc_1k_v3_fastqs.tar


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# psuedoalignment and generation of count matrices
cd $main_path
mkdir -p $main_path/pbmc_1k_v3/

kb count --verbose \
-i $main_path/indices/human_lamanno.idx \
-g $main_path/indices/human_lamanno.t2g \
-x 10xv3 \
-o $main_path/pbmc_1k_v3/ \
-t $threads -m 30G \
-c1 $main_path/indices/human_lamanno.mature.t2c \
-c2 $main_path/indices/human_lamanno.nascent.t2c \
--workflow lamanno --filter bustools --overwrite --loom \
$main_path/pbmc_1k_v3_raw/pbmc_1k_v3_fastqs/pbmc_1k_v3_S1_L001_R1_001.fastq.gz \
$main_path/pbmc_1k_v3_raw/pbmc_1k_v3_fastqs/pbmc_1k_v3_S1_L001_R2_001.fastq.gz
$main_path/pbmc_1k_v3_raw/pbmc_1k_v3_fastqs/pbmc_1k_v3_S1_L002_R1_001.fastq.gz \
$main_path/pbmc_1k_v3_raw/pbmc_1k_v3_fastqs/pbmc_1k_v3_S1_L002_R2_001.fastq.gz
