This is the implementation of 2D UNet. The code only covers the inference stage.


0. Prepare the deep learning environments

    a. Please install python==3.7, pytorch==1.4.0

    b. Please install CUDA10

1. Prepare the datasets

    a. Please preprocess your dicom to npydata, each CT volume needs to be resized to 368x368, the output npydata shape should be TxHxW.

    b. Please create the list file of CT volumes.

    c. Move the CT volumes to "NCOV-BF/NpyData" and the list file to "NCOV-BF/ImageSets". We have provided an example file in these directories.

2. Do evaluation

    Before running the command, make sure your variables in the cfgs directory
    CUDA_VISIBLE_DEVICES=0 python3 test.py cfgs/test.yaml
