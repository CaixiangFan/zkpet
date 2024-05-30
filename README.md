# zkpet
zkPET enables privacy-preserving and smart peer-to-peer energy trading via ZKML.

We simplify a set of simple linear models for long-term time series forcasting called [LTSF-Linear](https://github.com/cure-lab/LTSF-Linear), which were evaluated to surprisingly outperform existing sophisticated Transformer-based LTSF models. The simplified models are trained against the electricity dataset for a demand forcasting task. These trained models are then ZK-ed to generate a proof off-chain and verify on-chain. Performance evaluation results demonstrated the effectiveness of the proposed privacy-preserving peer-to-peer energy trading solution based on zero-knowledge machine learning.

# Steps of zkPET

## Train ML Models
1. Clone this repository, in the root, edit and execute the `train.sh` script to train the models in the background.
```
sh train.sh 
```
2. All trained models are saved in the `checkpoints` folder named by the hyperparameters. Saved files are `checkpoint.onnx` and `checkpoint.pth`. In this step, `checkpoint.onnx` is saved from `checkpoint.pth` with some dummy data. So, it cannot be directly used in ZKML.

## ZKML Inference Models
1. Load the `.pth` file and corresponding `electricity.csv` data to convert it to `.onnx` model. Execute the first cell in the `src/electricity_decentralized_ZKML.ipynb` file.
2. Change back to root, ZKML and profile the inference by running the following script.
```
sh src/electricity_decentralized_ZKML_single.sh
```
3. Plot the performance figures through `plot.ipynb` with `perf.txt` log files in the checkpoint folders.
