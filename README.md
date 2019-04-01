# CubiCasa5K: A Dataset and an Improved Multi-Task Model for Floorplan Image Analysis

## Multi-Task Model
The model uses the neural network architecture of previous work [Raster-to-Vector: Revisiting Floorplan Transformation](https://github.com/art-programmer/FloorplanTransformation). We ported the network from Torch to PyTorch. The pre- and post-processing is modified to suit our dataset, but we use some of the same code when possible. Also we use a multi-task uncertainty loss function introduced by [Kendall et al.](https://arxiv.org/abs/1705.07115) An example of the model's result can be found in the samples.ipynb file.

## Dataset
CubiCasa5K is a large-scale floorplan image dataset containing 5000 samples annotated into over 80 floorplan object categories. The dataset annotations are performed in a dense and versatile manner by using polygons for separating the different objects. You can download the CubiCasa5K dataset from [here](https://zenodo.org/deposit/2613548) and extract the zip file to data/ folder.

## Requirements
The model is written for Python 3.6.5 and Pytorch 1.0.0 with CUDA enabled GPU. Other dependencies Python can be found in requirements.txt file with the exception of cv2 3.1.0 ([OpenCV](https://opencv.org/)). If you want to use the Dockerfile you need to have docker and [nvidia-docker2](https://github.com/NVIDIA/nvidia-docker) installed. We use pre-built image [anibali/pytorch:cuda-9.0](https://github.com/anibali/docker-pytorch) as a starting point and install other required libraries using pip. To create the container run in the:
```bash
docker build -t cubi -f Dockerfile .
```
To start JupyterLab in the container:
```bash
docker run --rm -it --init \\
  --runtime=nvidia \\
  --ipc=host \\
  --publish 1111:1111 \\
  --user="$(id -u):$(id -g)" \\
  --volume=$PWD:/app \\
  -e NVIDIA_VISIBLE_DEVICES=0 \\
  cubi jupyter-lab --port 1111 --ip 0.0.0.0 --no-browser
```
You can now open a terminal in [JupyterLab web interface](http://localhost:1111) to execute more commands in the container.

## Database creation
We create a LMDB database of the dataset, where we store the floorplan image, segmentation tensors and heatmap coordinates. This way we can access the data faster during training and evaluation. The downside however is that the database takes about 105G of hard drive space. There is an option to parse the SVG file on the go but it is slow for training.
Commands to create the database:
```bash
python create_lmdb.py --txt val.txt
python create_lmdb.py --txt test.txt
python create_lmdb.py --txt train.txt
```

## Train
```bash
python train.py
```
Different training options can be found in the script file. Tensorboard is not included in the docker container. You need to run it outside and point it to cubi_runs/ folder. For each run a new folder is created with a timestamp as the folder name.
```bash
tensorboard --logdir runs_cubi/
```
## Evaluation
Our model weights file can be downloaded here. Once the weights file is in the project folder evaluation can be done. Also you can run the jupyter notebook file to see how the model is performing for different floorplans.
```bash
python eval.py --weights model_best_val_loss_var.pkl
```
Additional option for evaluation can be found in the script file. The results can be found in runs_cubi/ folder. 
