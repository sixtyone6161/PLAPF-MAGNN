

# PLAPF-MAGNN: Protein-Ligand Affinity Prediction Framework based on Molecular Characteristics and Automated Interactive Graph Neural Network

## Requirements

```setup
python==3.12
torch==2.4.1
CUDA==124
torch-cluster==1.6.3         
torch-geometric==2.6.1                 
torch-scatter==2.1.2
torch-sparse==0.6.18
```

## Dataset

> http://www.pdbbind.org.cn

## Training

To train the model(s) in the paper, run this command:

```train
just run the train.py
```



## Results

Our model achieves the following performance on CASF2016 :

| Model name    | RMSE  | Rp    |
| ------------- | ----- | ----- |
| 1_PLAPF-MAGNN | 0.908 | 0.917 |
| 2_PLAPF-MAGNN | 0.941 | 0.811 |
| 3_PLAPF-MAGNN | 0.882 | 0.835 |
|               |       |       |

