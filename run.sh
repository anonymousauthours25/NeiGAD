#!/bin/bash


dataset='books'
gpu=1

python train.py --gpu $gpu --dataset $dataset --model 'MLPAE' --nhidden 16 \
                                --pe_method 'adj' --pe_dim 17 --feat_norm 'none' --nlayer 3
python train.py --gpu $gpu --dataset $dataset --model 'GCNAE' --nhidden 16 \
                                --pe_method 'adj' --pe_dim 13 --feat_norm 'none' --nlayer 4
python train.py --gpu $gpu --dataset $dataset --model 'DOMINANT' --nhidden 16 \
                                --pe_method 'adj' --pe_dim 16 --feat_norm 'none' --nlayer 4
python train.py --gpu $gpu --dataset $dataset --model 'AnomalyDAE' --nhidden 16 \
                                --pe_method 'adj' --pe_dim 8 --feat_norm 'none' --nlayer 2
python train.py --gpu $gpu --dataset $dataset --model 'COLA' --nhidden 32 \
                                --pe_method 'adj' --pe_dim 10 --feat_norm 'none' --nlayer 3
python train.py --gpu $gpu --dataset $dataset --model 'GAAN' --nhidden 128 \
                                --pe_method 'adj' --pe_dim 9 --feat_norm 'none' --nlayer 2
python train.py --gpu $gpu --dataset $dataset --model 'CONAD' --nhidden 16 \
                                --pe_method 'adj' --pe_dim 16 --feat_norm 'none' --nlayer 4
python train.py --gpu $gpu --dataset $dataset --model 'GADNR' --nhidden 16 \
                                --pe_method 'adj' --pe_dim 70 --feat_norm 'none' --nlayer 3

dataset='enron'

python train.py --gpu $gpu --dataset $dataset --model 'MLPAE' --nhidden 64 \
                                --pe_method 'adj' --pe_dim 60 --feat_norm 'none' --nlayer 4
python train.py --gpu $gpu --dataset $dataset --model 'GCNAE' --nhidden 32 \
                                --pe_method 'adj' --pe_dim 7 --feat_norm 'none' --nlayer 3
python train.py --gpu $gpu --dataset $dataset --model 'DOMINANT' --nhidden 32 \
                                --pe_method 'adj' --pe_dim 10 --feat_norm 'none' --nlayer 4
python train.py --gpu $gpu --dataset $dataset --model 'AnomalyDAE' --nhidden 16 \
                                --pe_method 'adj' --pe_dim 6 --feat_norm 'none' --nlayer 2
python train.py --gpu $gpu --dataset $dataset --model 'COLA' --nhidden 32 \
                                --pe_method 'adj' --pe_dim 6 --feat_norm 'none' --nlayer 2
python train.py --gpu $gpu --dataset $dataset --model 'GAAN' --nhidden 64 \
                                --pe_method 'adj' --pe_dim 60 --feat_norm 'none' --nlayer 4
python train.py --gpu $gpu --dataset $dataset --model 'CONAD' --nhidden 16 \
                                --pe_method 'adj' --pe_dim 3 --feat_norm 'none' --nlayer 4
python train.py --gpu $gpu --dataset $dataset --model 'GADNR' --nhidden 128 \
                                --pe_method 'adj' --pe_dim 80 --feat_norm 'none' --nlayer 2



dataset='cora'

python train.py --gpu $gpu --dataset $dataset --model 'MLPAE' --nhidden 32 \
                                --pe_method 'adj' --pe_dim 15 --feat_norm 'none' --nlayer 4
python train.py --gpu $gpu --dataset $dataset --model 'GCNAE' --nhidden 16 \
                                --pe_method 'adj' --pe_dim 12 --feat_norm 'none' --nlayer 4
python train.py --gpu $gpu --dataset $dataset --model 'DOMINANT' --nhidden 128 \
                                --pe_method 'adj' --pe_dim 11 --feat_norm 'none' --nlayer 3
python train.py --gpu $gpu --dataset $dataset --model 'AnomalyDAE' --nhidden 128 \
                                --pe_method 'adj' --pe_dim 6 --feat_norm 'none' --nlayer 2
python train.py --gpu $gpu --dataset $dataset --model 'COLA' --nhidden 128 \
                                --pe_method 'adj' --pe_dim 12 --feat_norm 'none' --nlayer 2
python train.py --gpu $gpu --dataset $dataset --model 'GAAN' --nhidden 128 \
                                --pe_method 'adj' --pe_dim 3 --feat_norm 'none' --nlayer 4
python train.py --gpu $gpu --dataset $dataset --model 'CONAD' --nhidden 128 \
                                --pe_method 'adj' --pe_dim 11 --feat_norm 'none' --nlayer 3
python train.py --gpu $gpu --dataset $dataset --model 'GADNR' --nhidden 128 \
                                --pe_method 'adj' --pe_dim 4 --feat_norm 'none' --nlayer 3


dataset='citeseer'

python train.py --gpu $gpu --dataset $dataset --model 'MLPAE' --nhidden 128 \
                                --pe_method 'adj' --pe_dim 16 --feat_norm 'none' --nlayer 4
python train.py --gpu $gpu --dataset $dataset --model 'GCNAE' --nhidden 16 \
                                --pe_method 'adj' --pe_dim 5 --feat_norm 'none' --nlayer 4
python train.py --gpu $gpu --dataset $dataset --model 'DOMINANT' --nhidden 128 \
                                --pe_method 'adj' --pe_dim 6 --feat_norm 'none' --nlayer 4
python train.py --gpu $gpu --dataset $dataset --model 'AnomalyDAE' --nhidden 16 \
                                --pe_method 'adj' --pe_dim 5 --feat_norm 'none' --nlayer 2
python train.py --gpu $gpu --dataset $dataset --model 'COLA' --nhidden 128 \
                                --pe_method 'adj' --pe_dim 11 --feat_norm 'none' --nlayer 3
python train.py --gpu $gpu --dataset $dataset --model 'GAAN' --nhidden 128 \
                                --pe_method 'adj' --pe_dim 15 --feat_norm 'none' --nlayer 4
python train.py --gpu $gpu --dataset $dataset --model 'CONAD' --nhidden 128 \
                                --pe_method 'adj' --pe_dim 8 --feat_norm 'none' --nlayer 2
python train.py --gpu $gpu --dataset $dataset --model 'GADNR' --nhidden 128 \
                                --pe_method 'adj' --pe_dim 8 --feat_norm 'none' --nlayer 2

dataset='pubmed'

python train.py --gpu $gpu --dataset $dataset --model 'MLPAE' --nhidden 16 \
                                --pe_method 'adj' --pe_dim 60 --feat_norm 'none' --nlayer 3
python train.py --gpu $gpu --dataset $dataset --model 'GCNAE' --nhidden 64 \
                                --pe_method 'adj' --pe_dim 50 --feat_norm 'none' --nlayer 2
python train.py --gpu $gpu --dataset $dataset --model 'DOMINANT' --nhidden 16 \
                                --pe_method 'adj' --pe_dim 30 --feat_norm 'none' --nlayer 2
python train.py --gpu $gpu --dataset $dataset --model 'AnomalyDAE' --nhidden 128 \
                                --pe_method 'adj' --pe_dim 50 --feat_norm 'none' --nlayer 3
python train.py --gpu $gpu --dataset $dataset --model 'COLA' --nhidden 16 \
                                --pe_method 'adj' --pe_dim 60 --feat_norm 'none' --nlayer 2
python train.py --gpu $gpu --dataset $dataset --model 'GAAN' --nhidden 128 \
                                --pe_method 'adj' --pe_dim 40 --feat_norm 'none' --nlayer 2
python train.py --gpu $gpu --dataset $dataset --model 'CONAD' --nhidden 128 \
                                --pe_method 'adj' --pe_dim 30 --feat_norm 'none' --nlayer 3
python train.py --gpu $gpu --dataset $dataset --model 'GADNR' --nhidden 128 \
                                --pe_method 'adj' --pe_dim 30 --feat_norm 'none' --nlayer 3


dataset='acm'

python train.py --gpu $gpu --dataset $dataset --model 'MLPAE' --nhidden 64 \
                                --pe_method 'adj' --pe_dim 80 --feat_norm 'none' --nlayer 2
python train.py --gpu $gpu --dataset $dataset --model 'GCNAE' --nhidden 128 \
                                --pe_method 'adj' --pe_dim 100 --feat_norm 'none' --nlayer 2
python train.py --gpu $gpu --dataset $dataset --model 'DOMINANT' --nhidden 64 \
                                --pe_method 'adj' --pe_dim 80 --feat_norm 'none' --nlayer 3
python train.py --gpu $gpu --dataset $dataset --model 'AnomalyDAE' --nhidden 128 \
                                --pe_method 'adj' --pe_dim 100 --feat_norm 'none' --nlayer 2
python train.py --gpu $gpu --dataset $dataset --model 'COLA' --nhidden 16 \
                                --pe_method 'adj' --pe_dim 12 --feat_norm 'none' --nlayer 3
python train.py --gpu $gpu --dataset $dataset --model 'GAAN' --nhidden 16 \
                                --pe_method 'adj' --pe_dim 100 --feat_norm 'none' --nlayer 2
python train.py --gpu $gpu --dataset $dataset --model 'CONAD' --nhidden 32 \
                                --pe_method 'adj' --pe_dim 80 --feat_norm 'none' --nlayer 2
python train.py --gpu $gpu --dataset $dataset --model 'GADNR' --nhidden 128 \
                                --pe_method 'adj' --pe_dim 80 --feat_norm 'none' --nlayer 2

dataset='blogcatalog'

python train.py --gpu $gpu --dataset $dataset --model 'MLPAE' --nhidden 16 \
                                --pe_method 'adj' --pe_dim 12 --feat_norm 'none' --nlayer 4
python train.py --gpu $gpu --dataset $dataset --model 'GCNAE' --nhidden 128 \
                                --pe_method 'adj' --pe_dim 100 --feat_norm 'none' --nlayer 2
python train.py --gpu $gpu --dataset $dataset --model 'DOMINANT' --nhidden 128 \
                                --pe_method 'adj' --pe_dim 5 --feat_norm 'none' --nlayer 2
python train.py --gpu $gpu --dataset $dataset --model 'AnomalyDAE' --nhidden 16 \
                                --pe_method 'adj' --pe_dim 50 --feat_norm 'none' --nlayer 2
python train.py --gpu $gpu --dataset $dataset --model 'COLA' --nhidden 64 \
                                --pe_method 'adj' --pe_dim 9 --feat_norm 'none' --nlayer 3
python train.py --gpu $gpu --dataset $dataset --model 'GAAN' --nhidden 16 \
                                --pe_method 'adj' --pe_dim 30 --feat_norm 'none' --nlayer 2
python train.py --gpu $gpu --dataset $dataset --model 'CONAD' --nhidden 128 \
                                --pe_method 'adj' --pe_dim 8 --feat_norm 'none' --nlayer 2
python train.py --gpu $gpu --dataset $dataset --model 'GADNR' --nhidden 128 \
                                --pe_method 'adj' --pe_dim 3 --feat_norm 'none' --nlayer 2

dataset='dblp'

python train.py --gpu $gpu --dataset $dataset --model 'MLPAE' --nhidden 16 \
                                --pe_method 'adj' --pe_dim 3 --feat_norm 'none' --nlayer 4
python train.py --gpu $gpu --dataset $dataset --model 'GCNAE' --nhidden 16 \
                                --pe_method 'adj' --pe_dim 10 --feat_norm 'none' --nlayer 4
python train.py --gpu $gpu --dataset $dataset --model 'DOMINANT' --nhidden 64 \
                                --pe_method 'adj' --pe_dim 14 --feat_norm 'none' --nlayer 3
python train.py --gpu $gpu --dataset $dataset --model 'AnomalyDAE' --nhidden 16 \
                                --pe_method 'adj' --pe_dim 11 --feat_norm 'none' --nlayer 2
python train.py --gpu $gpu --dataset $dataset --model 'COLA' --nhidden 64 \
                                --pe_method 'adj' --pe_dim 5 --feat_norm 'none' --nlayer 4
python train.py --gpu $gpu --dataset $dataset --model 'GAAN' --nhidden 128 \
                                --pe_method 'adj' --pe_dim 11 --feat_norm 'none' --nlayer 4
python train.py --gpu $gpu --dataset $dataset --model 'CONAD' --nhidden 128 \
                                --pe_method 'adj' --pe_dim 20 --feat_norm 'none' --nlayer 2
python train.py --gpu $gpu --dataset $dataset --model 'GADNR' --nhidden 128 \
                                --pe_method 'adj' --pe_dim 11 --feat_norm 'none' --nlayer 3



