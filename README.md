# Deep Learning-based Detection for COVID-19 from Chest CT using Weak Label

By Chuansheng Zheng, Xianbo Deng, Qing Fu, Qiang Zhou, Hui Ma, Wenyu Liu, and Xinggang Wang.

<hr>

Before running the code, please prepare a computer with NVIDIA GPU. Then install anaconda, pytorch and NVIDIA CUDA driver. Then you can step into the two folders to check the `README.md` file.

- In the directory of "2dunet", the code mainly aims to segment the lung region to obtain all lung masks.
- In the directory of "deCoVnet", the code does the classification task of whether a CT volume being infected.

- The file "20200212-auc95p9.txt" contains the output probabilities of our pretrained deCovNet on our testing set.

The pretrained models are not currently available. If you are interested in the training codes, please contact [Xinggang Wang](mailto:xgwang@hust.edu.cn).

<hr>

We also provide online testing website for evaluating a CT volume whether being infected. Click [here](http://39.100.61.27) for more details.


# LICENSE

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

This work is licensed under a
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.

You should have received a copy of the license along with this
work. If not, see <http://creativecommons.org/licenses/by-nc-sa/4.0/>.


