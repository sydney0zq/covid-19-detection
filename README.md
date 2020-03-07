# Deep Learning-based Detection for 2019-Coronavirus Disease (COVID-19) from Chest Computed Tomography

This is the official implementation of the paper "Deep Learning-based Detection for 2019-Coronavirus Disease (COVID-19) from Chest Computed Tomography". It covers only the inference stage.

Before running the code, please prepare a computer with NVIDIA GPU. Then install anaconda, pytorch and NVIDIA CUDA driver. Then you can step into the two folders to check the `README.md` file.

- In the directory of "2dunet", the code mainly aims to segment the lung region to obtain all lung masks.
- In the directory of "deCoVnet", the code does the classification task of whether a CT volume being infected.

- The file "20200212-auc95p9.txt" contains the output probabilities of our pretrained deCovNet on our testing set.

Considering the privacy of patients, the training and testing data will not be publicly available. If you are interested about running the code, please contact the corresponding author [Xinggang Wang](mailto:xgwang@hust.edu.cn) for the pretrained models.



# LICENSE

This work is licensed under a [Creative Commons Attribution-ShareAlike 4.0
International License][cc-by-sa].

[![CC BY-SA 4.0][cc-by-sa-image]][cc-by-sa]

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg

