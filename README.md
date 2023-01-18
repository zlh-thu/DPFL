# DPFL
This is the official implementation of our paper [A Fine-grained Differentially Private Federated Learning against Leakage from Gradients](https://ieeexplore.ieee.org/abstract/document/9627872), accepted by IEEE Internet of Things Journal, 2021. This research project is developed based on Python 3 and Pytorch, created by [Linghui Zhu](https://github.com/zlh-thu).


## Reference
If our work or this repo is useful for your research, please cite our paper as follows:
```
@ARTICLE{9627872,
  author={Zhu, Linghui and Liu, Xinyi and Li, Yiming and Yang, Xue and Xia, Shu-Tao and Lu, Rongxing},
  journal={IEEE Internet of Things Journal}, 
  title={A Fine-Grained Differentially Private Federated Learning Against Leakage From Gradients}, 
  year={2022},
  volume={9},
  number={13},
  pages={11500-11512},
  doi={10.1109/JIOT.2021.3131258}}

```

## Pipeline
![Pipeline](https://github.com/zlh-thu/DPFL/blob/main/img/pipeline.png)


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```
Make sure the directory follows:
```File Tree
stealingverification
├── data
│   ├── cifar10
│   └── ...
├── ckpt 
│   
├── pogz
│   
├── model
|
```


## PoGZ
Load a pretrained local model and calculate the PoGZ of each layer with the local valid dataset.


