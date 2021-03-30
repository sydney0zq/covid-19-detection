# A Weakly-supervised Framework for COVID-19 Classification and Lesion Localization from Chest CT

By Xinggang Wang, Xianbo Deng, Qing Fu, Qiang Zhou, Jiapei Feng, Hui Ma, Wenyu Liu, Chuansheng Zheng.

<hr>

This project aims at providing a deep learning algorithm to detect COVID-19 from chest CT using weak label. And the souce code of **training and testing** is provided. If you have interests about more details, please check [our paper](http://doi.org/10.1109/TMI.2020.2995965) (IEEE Transactions on Medical Imaging). 

<hr>

Before running the code, please prepare a computer with NVIDIA GPU, then install Anaconda, PyTorch and NVIDIA CUDA driver. Once the environment and dependent libraries are installed, please check the `README.md` files in `2dunet` and `deCoVnet` directories.

- In the directory of "2dunet", the code mainly aims to segment the lung region to obtain all lung masks.
- In the directory of "deCoVnet", the code does the classification task of whether a CT volume being infected.
- In the directory of "lesion\_loc", the code mainly implements the lesion localization.

- The file "20200212-auc95p9.txt" contains the output probabilities of our pretrained deCovNet on our testing set.

The pretrained models are currently available at Google Drive, [unet](https://drive.google.com/file/d/1tTnEm5lJPXdgOc5Vvot1vnupBIkvlcxz/view?usp=sharing) and [deCoVnet](https://drive.google.com/file/d/1ggA6rvzTWAPYhqB42qlvmsdSREGyu-5q/view?usp=sharing). 


If you have any other questions, please contact [Xinggang Wang](mailto:xgwang@hust.edu.cn).


# LICENSE

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

This work is licensed under a
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.

You should have received a copy of the license along with this
work. If not, see <http://creativecommons.org/licenses/by-nc-sa/4.0/>.


