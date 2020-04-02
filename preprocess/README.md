The `preprocess` directory contain two folders:

- `preprocess-obtain-lungmasks`: This directory segments the 3D lung masks from the CT volume by morphological algorithm and it is possible to fail. The obtained lung masks are provided as the ground-truth when training our 2dunet. Please remove out the failed cases by sorting the size of the obtained lung masks.
- `preprocess-obtain-puredicom`: This directory converts your dicom CT volume into a standard numpy format with the shape of TxHxW. And we will input the numpy data into the network for training and testing.
