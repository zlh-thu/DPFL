# DPFL
This is the official implementation of our paper [A Fine-grained Differentially Private Federated Learning against Leakage from Gradients](https://ieeexplore.ieee.org/abstract/document/9627872), accepted by IEEE Internet of Things Journal, 2021. This research project is developed based on Python 3 and Pytorch.


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
The PoGZ
```
python get_pogz.py --dataset=dataset_name --resume_path=./ckpt/path_to_pretrained_model.pt --local_val_dataset_path=./path_to_local_val_dataset/
```
The result will be saved in ./pogz/ .

## Add noise on client model
Load a updated client model and add noised.
```
python add_noise.py --resume_path=./ckpt/path_to_updated_local_model.pt --dataset=dataset_name 
```


