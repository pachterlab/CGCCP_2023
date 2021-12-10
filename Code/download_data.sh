#### Download

mkdir data

# Download and process 10x

wget --content-disposition "https://data.caltech.edu/tindfiles/serve/6b5d2a60-1e86-4017-9506-51c0132d4172/"

mv raw_loom.tar data/
cd data
tar -xvf raw_loom.tar
cd ..

## Download and process 10X protein and mRNA counts

wget --content-disposition "https://figshare.com/ndownloader/files/17820449"
wget --content-disposition "https://figshare.com/ndownloader/files/17820452"
wget --content-disposition "https://figshare.com/ndownloader/files/17826998"
wget --content-disposition "https://figshare.com/ndownloader/files/17820455"
gunzip pbmc_10x_10k_fbc.loom.gz
gunzip pbmc_1x_10k_fbc.loom.gz

mv pbmc* data/loom_10x_kb/

##
wget --content-disposition "http://pklab.med.harvard.edu/velocyto/hgForebrainGlut/hgForebrainGlut.loom"
mv hgForebrainGlut.loom data/
