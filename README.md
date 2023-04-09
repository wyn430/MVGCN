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

The code has been tested on following environment

```
Ubuntu 18.04
python 3.6
CUDA 11.0
torch 1.4.0
scikit-learn 0.21.3
geomloss 0.2.3
```

## Dataset
The experiments are conducted on synthetic and real-scanned 3D point cloud. The synthetic data is available at [folder](). To run the code directly, please download and correctly set the directory of data.

## Usage

### MVGCN

Train the proposed MVGCN

CUDA_VISIBLE_DEVICES=ID python main.py --model [model name] --k [# of global neighbor] --kl [# of local neighbor] --batch_size [batch size] --dataset [directory of dataset] --output_C [# of class]

```
e.g. CUDA_VISIBLE_DEVICES=0 python main.py --model hgcnn --k 15 --kl 15 --batch_size 42 --dataset ./Data/Surface_Defects_pcd_extend_2000_estnorm_noise0001 --output_C 4
```


## License

This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/wyn430/MVGCN/blob/master/LICENSE) file for details.

## Acknowledgments

The implementation of graph convolution operation [repo](https://github.com/WangYueFt/dgcnn).
