# sane-annotation-shape-completion (WIP)

## Environment
Tested on Debian 9.9, Cuda: 10.0, Python: 3.6, Pytorch: 1.2.0 with Anaconda

## Installation
```
git clone --recursive https://github.com/ziliHarvey/smart-annotation-pointrcnn.git
cd app/PointCNN/
sh build_and_install.sh
```
If you are using anaconda, install the environment dependencies by using environment.yaml

Testing file is at app/rl_gan/test_rl_module.py

## Usage

**Works on app/test_dataset/0_drive_0064_sync/sample/argoverse/lidar**
```
cd app
python app.py
```
And open your browser and go to http://0.0.0.0:7772.

Testing  
- Draw an approximate bounding box in one-click by pressing "a" and clicking anywhere near object
- Tick over option "Point Completion" on the left button panel to get extra points to complete the pointcloud
- Tick over option "Shape Completion" to complete the points and get a shape of object using Convex-hulling

Debugging  
- Use pdb
- Use visdom provided to display output plots of completion

<p align="center">
  <img width="860" height="400" src="media/point_complete.gif">
</p>

### Progress
- [x] Adding PointCNN Segmentation model
- [x] Adding PointRCNN Box regresssion backend
- [x] One-click box fitting
- [x] Segmented object points display
- [x] Incorporate RL-GAN-Net to do point-completion (Inference) (Training using Car Shapenet-models)
- [x] Upgrade Encode decoder to PointCompletion Network to get robust Point Completion
- [x] Display Shape completion with Convex hulling
- [x] Adding Kalman filter tracking 
- [ ] Use Deep learning model to do direct shape completion with image & Pointcloud
