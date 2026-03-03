
### Official PyTorch implementation for the HGTS-Former

* **HGTS-Former: Hierarchical HyperGraph Transformer for Multivariate Time Series Analysis**, 
  Xiao Wang, Member, IEEE, Hao Si, Fan Zhang, Xiaoya Zhou, Dengdi Sun, Wanli Lyu, Qingquan Yang, Jin Tang
  [[Paper]()]
  [[Code]()]

##  Overview

HGTS-Former is a novel hypergraph-based backbone for multivariate time series analysis, which addresses complex variable coupling by constructing hierarchical hypergraphs, aggregates latent temporal patterns within and across channels via a sparse attention mechanism, and adaptively updates node representations through an EdgeToNode module.
![](../figure/hypergraphV3.jpg)

## Requirements

Please ensure that you are using Python 3.11.0 and install the required dependencies.
```
torch = 2.0.1+cu118
einops = 0.8.0
matplotlib = 3.9.2
numpy = 1.25.0
pandas = 1.5.3
scikit-learn = 1.2.2
transformers = 4.40.1
```

## Prepare Datastes

All datasets can be obtained from [[Google Drive]](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2?usp=sharing), [[Baidu Drive]](https://pan.baidu.com/s/1r3KhGd0Q9PJIUZdfEYoymg?pwd=i9iy) or [[Hugging Face]](https://huggingface.co/datasets/thuml/Time-Series-Library)

```
data
└── electricity
	└──electricity.csv
└── ETT-small
    └── ETTh1.csv
    └── ETTh2.csv
    └── ETTm1.csv
    └── ETTm2.csv
└── traffic
	└── traffic.csv
└── weather
	└── weather.csv
└── solar
	└── solar_AL.txt
```
## Train & Test

All scripts are located in `./scripts`. For instance, to train or test a model using the ETTh1 dataset with an input length of 672, simply run:
```shell
bash ./scripts/ETTh1.sh
```

## Source code 

* [[**OpenLTM**](https://github.com/thuml/OpenLTM)]  is an open codebase aiming to provide a pipeline to develop and evaluate large time-series models.

* [[**Time-Series-Library**](https://github.com/thuml/Time-Series-Library)] is an open-source library for deep learning researchers, especially for deep time series analysis.

## Citation
If you find this repo useful, please consider citing our paper as follows:
```bibtex
@article{wang2025hgts,
  title={HGTS-Former: Hierarchical HyperGraph Transformer for Multivariate Time Series Analysis},
  author={Wang, Xiao and Si, Hao and Zhang, Fan and Zhou, Xiaoya and Sun, Dengdi and Lyu, Wanli and Yang,Qingquan and Tang, Jin},
  journal={arXiv preprint arXiv:2508.02411},
  year={2025}
}
```
  




  


