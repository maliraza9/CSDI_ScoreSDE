# Score-Based Stochastic Stochastic Differential Equations Implementation of Score Matching for Irregular Time Series
This is the github repository for the Score SDE algorithms Variance Preserving SDE and Variance Exploding SDE within the 
CSDI framework implementation of the thesis Score Matching for Irregularly Sampled Time Series.

## Requirement

Please install the packages in requirements.txt

## Preparation
### Download the healthcare dataset 
```shell
python download.py physio
```
### Download the air quality dataset 
```shell
python download.py pm25
```

## Experiments 

### training and imputation for the healthcare dataset
```shell
python exe_physio.py --testmissingratio [missing ratio] --nsample [number of samples]
```

### imputation for the healthcare dataset with pretrained model
```shell
python exe_physio.py --modelfolder pretrained --testmissingratio [missing ratio] --nsample [number of samples]
```

### training and imputation for the healthcare dataset
```shell
python exe_pm25.py --nsample [number of samples]
```

### Visualize results
'visualize_examples.ipynb' is a notebook for visualizing results.

## Acknowledgements

A part of the codes is based on [BRITS](https://github.com/caow13/BRITS) and [DiffWave](https://github.com/lmnt-com/diffwave)

## Citation
If you use this code for your research, please cite our paper:

```
@inproceedings{tashiro2021csdi,
  title={CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation},
  author={Tashiro, Yusuke and Song, Jiaming and Song, Yang and Ermon, Stefano},
  booktitle={Advances in Neural Information Processing Systems},
  year={2021}
}
```
