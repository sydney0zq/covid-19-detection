# A Weakly-supervised Framework for COVID-19 Classification and Lesion Localization from Chest CT

By Xinggang Wang, Xianbo Deng, Qing Fu, Qiang Zhou, Jiapei Feng, Hui Ma, Wenyu Liu, Chuansheng Zheng.

<hr>

This project aims at providing a deep learning algorithm to detect COVID-19 from chest CT using weak label. And the souce code of **training and testing** is provided. If you have interests about more details, please check [our paper](http://doi.org/10.1109/TMI.2020.2995965) (IEEE Transactions on Medical Imaging). 


**Note: We provide an online testing website for evaluating whether a CT volume being infected, click [here](http://39.100.61.27) to test your own chest CT.**

<hr>

Before running the code, please prepare a computer with NVIDIA GPU, then install Anaconda, PyTorch and NVIDIA CUDA driver. Once the environment and dependent libraries are installed, please check the `README.md` files in `2dunet` and `deCoVnet` directories.

- In the directory of "2dunet", the code mainly aims to segment the lung region to obtain all lung masks.
- In the directory of "deCoVnet", the code does the classification task of whether a CT volume being infected.
- In the directory of "lesion\_loc", the code mainly implements the lesion localization.

- The file "20200212-auc95p9.txt" contains the output probabilities of our pretrained deCovNet on our testing set.

The pretrained models are not currently available. If you have any questions, please contact [Xinggang Wang](mailto:xgwang@hust.edu.cn).


# LICENSE

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

This work is licensed under a
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.

You should have received a copy of the license along with this
work. If not, see <http://creativecommons.org/licenses/by-nc-sa/4.0/>.


