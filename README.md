# A Lightweight Hybrid Model with Location-Preserving ViT for Efficient Food Recognition

The repository contains the source code for training the model proposed in the paper for food image recognition tasks.

## Installation

The source code can be obtained in the local python environment using the below command:

```git
git clone git@github.com:LduIIPLab/CVnets.git
cd CVnets/FoodProject
pip install -r requirements.txt
```

## Model Zoo

For the test results and model parameter files of the model proposed in this paper, please refer to [Model Zoo](README-model-zoo.md).

## Getting Started

```cmd
python main_train.py
```

- You can directly run the above code on the server terminal to train the model.
- Before training, you need to modify the path of datasets in config files for different datasets.
- If you want to train different data or change the size of the model, you can achieve this by modifying the code below in the main_train.py file.
  - "food101": ETHZ Food-101
  - "food172": Vireo Food-172
  - "food256": UEC Food-256


```python
experiments_config.ehfr_net_config_forward(dataset_name="food101", width_multiplier=0.5)
```

The images should be stored in the following two formats:

```
Food-101/
├── train
│   ├── 1
│   │   ├── 1_2.JPEG
│   │   ├── 1_3.JPEG
...
├── val
│   ├── 1
│   │   ├── 1_6.JPEG
...
```

- We provide a dataset called "food" to read the image in the storage format above and modify the relevant parts in the config file like the code provided below

```yaml
dataset:
  root_train: "/ai35/Food/Food-101"
  root_val: "/ai35/Food/Food-101"
  train_index_file: "train_full.txt"
  val_index_file: "val_full.txt"
  name: "food"
```



```
Food-101/
├── apple_pie
│   ├── 134.JPEG
│   ├── 21063.JPEG
...
├── baby_back_ribs
│   ├── 2432.JPEG
...
├── train_full.txt
├── val_full.txt
```

- We provide another dataset called "food_annother" to read the image in the storage format above and modify the relevant parts in the config file like the code provided below

```yaml
dataset:
  root_train: "/ai35/Food/Food-101"
  root_val: "/ai35/Food/Food-101"
  name: "food_another"
```

- If you want to train your own dataset, please put the format of the dataset into the folder according to the two formats provided above, and generate a new yaml file for configuring the config. The name format is "model name_dataset file name. yaml", and make the relevant configuration.
- In addition, it is necessary to add the code for training one's own model in the form of internal code in "experiments.config. py", and modify the code at the corresponding position in "main_train. py".

## Citation

If you find our project is helpful, please feel free to leave a star and cite our paper:

```
@article{sheng2024EHFR-Net
	title={A Lightweight Hybrid Model with Location-Preserving ViT for Efficient Food Recognition},
	author={Guorui Sheng, Weiqing Min, Xiangyi Zhu, Liang Xu, Qingshuo Sun, Yancun Yang, Lili Wang and Shuqiang Jiang},
	journal={Nutrients},
	year={2024}
}
```

## License