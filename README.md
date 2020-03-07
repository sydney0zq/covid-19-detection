# Deep Learning-based Detection for 2019-Coronavirus Disease (COVID-19) from Chest Computed Tomography

This is the official implementation of the paper "Deep Learning-based Detection for 2019-Coronavirus Disease (COVID-19) from Chest Computed Tomography". It covers only the inference stage.

Before running the code, please prepare a computer with NVIDIA GPU. Then install anaconda, pytorch and NVIDIA CUDA driver. Then you can step into the two folders to check the `README.md` file.

- In the directory of "2dunet", the code mainly aims to segment the lung region to obtain all lung masks.
- In the directory of "deCoVnet", the code does the classification task of whether a CT volume being infected.

- The file "20200212-auc95p9.txt" contains the output probabilities of our pretrained deCovNet on our testing set.

Considering the privacy of patients, the training and testing data will not be publicly available. If you are interested about running the code, please contact the corresponding author [Xinggang Wang](mailto:xgwang@hust.edu.cn) for the pretrained models.



# LICENSE

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

This work is licensed under a
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.

You should have received a copy of the license along with this
work. If not, see <http://creativecommons.org/licenses/by-nc-sa/4.0/>.


