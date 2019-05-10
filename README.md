## SMPL-X:  A new, unified, 3D model of the human body 

[[Paper Page](https://smpl-x.is.tue.mpg.de/)] [[Paper](https://ps.is.tuebingen.mpg.de/uploads_file/attachment/attachment/497/SMPL-X.pdf)]

## Table of Contents
  * [Installation](#installation)
  * [Dowloading the model](#downloading-the-model)
  * [Example](#example)
  * [Citation](#citation)
  * [Contact](#contact)

## Description

*SMPL-X* (SMPL eXpressive) is a unified body model with shape parameters trained jointly for the
face, hands and body. *SMPL-X* uses standard vertex based linear blend skinning with learned corrective blend
shapes, has N = 10, 475 vertices and K = 54 joints,
which includes joints for the neck, jaw, eyeballs and fingers. 
SMPL-X is defined by a function M(θ, β, ψ), where θ is the pose parameter, β the shape parameter and
ψ the expression parameter.


## Installation

To install the model simply you can:
1. To install from PyPi simply run: 
  ```Shell
  pip install smplx
  ```
2. Clone this repository and install it using the *setup.py* script: 
```Shell
git clone https://github.com/vchoutas/smplx
python setup.py install`
```

## Downloading the model

To download the *SMPL-X* go to the [project website](https://smpl-x.is.tue.mpg.de/) and register to get access to the downloads section. 


## Example

After installing the *smplx* package and downloading the model parameters you should be able to run the *demo.py*
script to visualize the results. For this step you have to install the [pyrender](https://pyrender.readthedocs.io/en/latest/index.html) and [trimesh](https://trimsh.org/) packages.

`python examples/demo.py --model-folder $SMPLX_FOLDER --plot-joints=True --gender="neutral"`

![SMPL-X Examples](./images/example.png)

## Citation

```
@inproceedings{SMPL-X:2019,
  title = {Expressive Body Capture: 3D Hands, Face, and Body from a Single Image},
  author = {Pavlakos, Georgios and Choutas, Vasileios and Ghorbani, Nima and Bolkart, Timo and Osman, Ahmed A. A. and Tzionas, Dimitrios and Black, Michael J.},
  booktitle = {Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
  year = {2019}
}
```

## Contact
For questions about our paper or code, please contact [Vassilis Choutas](vassilis.choutas@tuebingen.mpg.de).
