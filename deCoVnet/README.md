This is the implementation of DeCoVNet. The code only covers training and testing stages.


0. Prepare the deep learning environments

    a. Please install python==3.7, pytorch==1.4.0

    b. Please install CUDA10

1. Prepare the datasets

    a. Please preprocess your data obtained in 2dunet directory, each CT volume needs to be cropped and resized to 224x336, the output npydata and npymask should be TxHxW. (patient_uid.npy & patient_uid-mask.npy) The script of cropping and resizing is `cropresize.py`, please check it for its usage.

    b. Please create the list file of CT volumes.

    c. Move the npydata and npymask to "NCOV-BF/NpyData-size224x336" and the list file to "NCOV-BF/ImageSets". We have provided an example file in these directories.

2. Do training and testing

    - Training: `CUDA_VISIBLE_DEVICES=0 python3 train.py cfgs/trainval.yaml`
    - Testing: `CUDA_VISIBLE_DEVICES=0 python3 test.py cfgs/test.yaml`


Note the final accuracy and AUC are about 90 and 96 on our validation set respectively (about 110 epoches), maybe you can stop your program when your model reaches them.
