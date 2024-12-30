
# NeiGAD

run the code for training Graph Anomaly Detection

```
python train.py --gpu $gpu --dataset $dataset --model 'MLPAE' --nhidden 32 --pe_method 'adj' --pe_dim 15 --feat_norm 'none' --nlayer 4

python train.py --gpu $gpu --dataset $dataset --model 'GCNAE' --nhidden 16 --pe_method 'adj' --pe_dim 12 --feat_norm 'none' --nlayer 4

python train.py --gpu $gpu --dataset $dataset --model 'DOMINANT' --nhidden 128 --pe_method 'adj' --pe_dim 11 --feat_norm 'none' --nlayer 3

python train.py --gpu $gpu --dataset $dataset --model 'AnomalyDAE' --nhidden 128 --pe_method 'adj' --pe_dim 6 --feat_norm 'none' --nlayer 2
```

or
```
./run.sh
```