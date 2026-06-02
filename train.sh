#train teacher DGM
cd exps/LVD_for_imagenet/
python3 train_vqvae2_model.py -id --data-path "$DATA_PATH"
#get LVD augmented dataset
python3 get_data_for_PG.py -id --data-path "$DATA_PATH" --num-tr-samples 60000 --num-ts-samples 10000
#train cluster-conditioned PCs with progressive growing
cd ../progressive_growing/
bash pg.sh "imagenet32"
#finetune PCs
cd ../LVD_for_imagenet/
python-jl progressive_growing_top.py -id --data-path "$DATA_PATH"


#specify the data path
# data_path=""
# #train teacher DGM
# cd exps/LVD_for_imagenet/
# python train_vqvae2_model.py -id -img 64 -p 8 --data-path $DATA_PATH
# #get LVD augmented dataset
# python get_data_for_PG.py -id -img 64 -p 8 --data-path $DATA_PATH --num-tr-samples 60000 --num-ts-samples 10000
# #train cluster-conditioned PCs with progressive growing
# cd ../progressive_growing/
# bash pg.sh "imagenet64"
# #finetune PCs
# cd ../LVD_for_imagenet/
# python-jl progressive_growing_top.py -id -img 64 -p 8 --data-path $DATA_PATH