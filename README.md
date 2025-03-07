# MVGCN

Implementation of our recent paper, [MVGCN: Multi-View Graph Convolutional Neural Network for Surface Defect Identification Using Three-Dimensional Point Cloud](https://asmedigitalcollection.asme.org/manufacturingscience/article/145/3/031004/1148268/MVGCN-Multi-View-Graph-Convolutional-Neural).

## Abstract
Surface defect identification is a crucial task in many manufacturing systems, including automotive, aircraft, steel rolling, and precast concrete. Although image-based surface defect identification methods have been proposed, these methods usually have two limitations: images may lose partial information, such as depths of surface defects, and their precision is vulnerable to many factors, such as the inspection angle, light, color, noise, etc. Given that a three-dimensional (3D) point cloud can precisely represent the multidimensional structure of surface defects, we aim to detect and classify surface defects using a 3D point cloud. This has two major challenges: (i) the defects are often sparsely distributed over the surface, which makes their features prone to be hidden by the normal surface and (ii) different permutations and transformations of 3D point cloud may represent the same surface, so the proposed model needs to be permutation and transformation invariant. In this paper, a two-step surface defect identification approach is developed to investigate the defects’ patterns in 3D point cloud data. The proposed approach consists of an unsupervised method for defect detection and a multi-view deep learning model for defect classification, which can keep track of the features from both defective and non-defective regions. We prove that the proposed approach is invariant to different permutations and transformations. Two case studies are conducted for defect identification on the surfaces of synthetic aircraft fuselage and the real precast concrete specimen, respectively. The results show that our approach receives the best defect detection and classification accuracy compared with other benchmark methods.

## Citation

If you find our work useful in your research, please consider citing:

```
@article{wang2023mvgcn,
  title={MVGCN: Multi-View Graph Convolutional Neural Network for Surface Defect Identification Using Three-Dimensional Point Cloud},
  author={Wang, Yinan and Sun, Wenbo and Jin, Jionghua and Kong, Zhenyu and Yue, Xiaowei},
  journal={Journal of Manufacturing Science and Engineering},
  volume={145},
  number={3},
  pages={031004},
  year={2023},
  publisher={American Society of Mechanical Engineers}
}
```

## Installation

The code has been tested on following two environments

```
Ubuntu 18.04
python 3.6
CUDA 11.8
torch 1.4.0
scikit-learn 0.21.3
```
and 

```
Ubuntu 20.04
python 3.10.10
CUDA 11.8
torch 2.0.0
scikit-learn 1.2.2
```

## Dataset
The experiments are conducted on synthetic and real-scanned 3D point cloud. The synthetic data is available at [folder](https://drive.google.com/drive/folders/11d5L1DJomhu_RCGtVBslECfsJv48o7AC?usp=sharing). To run the code directly, please download and correctly set the directory of data. The scripts of conducting clustering and calculating Geodesic distance is included in the Jupyter Notebook files.

## Usage

### MVGCN
CUDA_VISIBLE_DEVICES=ID python main.py --model [model name] --k [# of global neighbor] --kl [# of local neighbor] --batch_size [batch size] --dataset [directory of dataset] --output_C [# of class]

Train the proposed MVGCN
```
e.g. CUDA_VISIBLE_DEVICES=0 python main.py --model hgcnn --k 15 --kl 15 --batch_size 42 --dataset ./Data/Surface_Defects_pcd_extend_2000_estnorm_noise0001 --output_C 4
```
The available "model name" includes:

* hgcnn: MVGCN only uses coordinates of points and builds graph using Euclidean distance
* hgcnn_norm: MVGCN uses coordinates and normal vectors of points and builds graph using Euclidean distance
* dgcnn: Graph convolution neural network only uses coordinates of points and builds graph using Euclidean distance
* dgcnn_norm: Graph convolution neural network uses coordinates and normal vectors of points and builds graph using Euclidean distance
* pointnet: Pointnet only uses coordinates of points
* pointnet_norm: Pointnet uses coordinates and normal vectors of points

The above models uses dataset in folder "Surface_Defects_pcd_extend_2000_estnorm_noise0001"

* hgcnn_geo: MVGCN only uses coordinates of points and builds graph using Geodesic distance
* hgcnn_norm_geo: MVGCN uses coordinates and normal vectors of points and builds graph using Geodesic distance
* dgcnn_geo: Graph convolution neural network only uses coordinates of points and builds graph using Geodesic distance
* dgcnn_norm_geo: Graph convolution neural network uses coordinates and normal vectors of points and builds graph using Geodesic distance

The above models uses dataset in folder "Surface_Defects_pcd_extend_2000_geod_estnorm_noise0001"

The only difference between these two datasets is that the pairwise Geodesic distance is pre-calculated and put in "Surface_Defects_pcd_extend_2000_geod_estnorm_noise0001".

### Pretrained Models
Two of the pretrained models are in [folder](https://github.com/wyn430/MVGCN/tree/master/pretrained). It is trained and tested on the dataset "Surface_Defects_pcd_extend_2000_estnorm_noise0001".

## License

This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/wyn430/MVGCN/blob/master/LICENSE) file for details.

## Acknowledgments

The implementation of graph convolution operation refers to [repo](https://github.com/WangYueFt/dgcnn).
