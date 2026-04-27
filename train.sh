#specify the data path
data_path="../data"
#train teacher DGM
cd exps/LVD_for_imagenet/
python3 train_vqvae2_model.py -id --data-path "$data_path"
#get LVD augmented dataset
python3 get_data_for_PG.py -id --data-path "$data_path"
#train cluster-conditioned PCs with progressive growing
cd ../progressive_growing/
bash pg.sh "imagenet32"
#finetune PCs
cd ../LVD_for_imagenet/
python-jl progressive_growing_top.py -id --data-path "$data_path"


#specify the data path
# data_path=""
# #train teacher DGM
# cd exps/LVD_for_imagenet/
# python train_vqvae2_model.py -id -img 64 -p 8 --data-path $data_path
# #get LVD augmented dataset
# python get_data_for_PG.py -id -img 64 -p 8 --data-path $data_path
# #train cluster-conditioned PCs with progressive growing
# cd ../progressive_growing/
# bash pg.sh "imagenet64"
# #finetune PCs
# cd ../LVD_for_imagenet/
# python-jl progressive_growing_top.py -id -img 64 -p 8 --data-path $data_path