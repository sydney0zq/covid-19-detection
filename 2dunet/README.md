This is the implementation of 2D UNet. The code only covers the training and testing stages.


0. Prepare the deep learning environments

    a. Please install python==3.7, pytorch==1.4.0

    b. Please install CUDA10

1. Prepare the datasets

    a. Please preprocess your dicom to npydata, each CT volume needs to be resized to 512x512, the output npydata shape should be TxHxW.

    b. Please create the list file of CT volumes.

    c. Move the CT volumes to "NCOV-BF/NpyData" and the list file to "NCOV-BF/ImageSets". We have provided an example file in these directories.

2. Do training and testing

    - Training: `CUDA_VISIBLE_DEVICES=0 python3 train.py cfgs/trainval.yaml`
    - Testing: `CUDA_VISIBLE_DEVICES=0 python3 test.py cfgs/test.yaml`


Note for training, we run almost 130 epoches and the final mIOU on validation set is about 0.97. Maybe you can stop your training when your accuracy is close to it.
