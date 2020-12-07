# Model parameter transfer 

## Table of Contents
  * [License](#license)
  * [Description](#description)
  * [Using the code](#using-the-code)
    * [Data](#data)
    * [SMPL to SMPL-X](#smpl-to-smpl-x)
    * [SMPL-X to SMPL](#smpl-x-to-smpl)
    * [SMPL+H to SMPL](#smpl%2Bh-to-smpl)
    * [SMPL to SMPL+H](#smpl-to-smpl%2Bh)
    * [SMPL+H to SMPL-X](#smpl%2Bh-to-smpl-x)
    * [SMPL-X to SMPL+H](#smpl-x-to-smpl%2Bh)
  * [Visualize correspondences](visualize-correspondences)
  * [Citation](#citation)
  * [Acknowledgments](#acknowledgments)
  * [Contact](#contact)

## License

Software Copyright License for **non-commercial scientific research purposes**.
Please read carefully the [terms and conditions](https://github.com/vchoutas/smplx/blob/master/LICENSE) and any accompanying documentation before you download and/or use the SMPL-X/SMPLify-X model, data and software, (the "Model & Software"), including 3D meshes, blend weights, blend shapes, textures, software, scripts, and animations. By downloading and/or using the Model & Software (including downloading, cloning, installing, and any other use of this github repository), you acknowledge that you have read these terms and conditions, understand them, and agree to be bound by them. If you do not agree with these terms and conditions, you must not download and/or use the Model & Software. Any infringement of the terms of this agreement will automatically terminate your rights under this [License](./LICENSE).

## Description

The repository contains code for converting model parameters of one model to
another. **Never** copy parameters between the models. You will not get the
same poses. SMPL, SMPL+H and SMPL-X shape spaces are **NOT** compatible, since
each model is the result of a different training process.
A more detailed explanation on how we extract correspondences
between the models and the loss function used to estimate the parameters can be
found [here](./docs/transfer.md).

## Requirements

1. Start by cloning the SMPL-X repo:
```Shell 
git clone https://github.com/vchoutas/smplx.git
```
2. Run the following command to install all necessary requirements
```Shell
    pip install -r requirements.txt
```
3. Install the Torch Trust Region optimizer by following the instructions [here](https://github.com/vchoutas/torch-trust-ncg).


## Using the code

### Data

Register on the [SMPL-X website](http://smpl-x.is.tue.mpg.de/), go to the
downloads section to get the correspondences and sample data,
by clicking on the *Model correspondences* button.
Create a folder
named `transfer_data` and extract the downloaded zip there. You should have the
following folder structure now:

```bash
transfer_data
├── meshes
│   ├── smpl
│   ├── smplx
├── smpl2smplh_def_transfer.pkl
├── smpl2smplx_deftrafo_setup.pkl
├── smplh2smpl_def_transfer.pkl
├── smplh2smplx_deftrafo_setup.pkl
├── smplx2smpl_deftrafo_setup.pkl
├── smplx2smplh_deftrafo_setup.pkl
├── smplx_mask_ids.npy
```

### SMPL to SMPL-X

To run the code to convert SMPL meshes to SMPL-X parameters use the following command:
  ```Shell
  python -m transfer_model --exp-cfg config_files/smpl2smplx.yaml
  ```
This should be run from the top directory of the repository.

The file *smpl2smplx.yaml* contains a sample configuration that reads meshes from a folder,
processes them and returns pkl files with SMPL-X parameters. To run on your own data create a folder
with SMPL meshes, in either ply or obj format, change the path in the config file and run the code.

### SMPL-X to SMPL

To run the code to convert SMPL-X meshes to SMPL parameters use the following command:
  ```Shell
  python main.py --exp-cfg config_files/smplx2smpl.yaml
  ```

The file *smplx2smpl.yaml* contains a sample configuration that reads meshes from a folder,
processes them and returns pkl files with SMPL parameters. To run on your own data create a folder
with SMPL-X meshes, in either ply or obj format, change the path in the config file and run the code.
When creating the SMPL-X meshes, do not use the hand and face parameters. 
Naturally, you will lose all hand and face information if you choose this, since
SMPL cannot model them.


### SMPL+H to SMPL

To run the code to convert SMPL+H meshes to SMPL parameters use the following command from the root `smplx` directory:
  ```Shell
  python -m transfer_model --exp-cfg config_files/smplh2smpl.yaml
  ```
This should be run from the top directory of the repository.

The file *smplh2smpl.yaml* contains a sample configuration that reads meshes from a folder,
processes them and returns pkl files with SMPL parameters. To run on your own data create a folder
with SMPL+H meshes, in either ply or obj format, change the path in the config file and run the code.
Note that using this direction means that you will lose information on the
hands.


### SMPL to SMPL+H

To run the code to convert SMPL meshes to SMPL+H parameters use the following command:
  ```Shell
  python -m transfer_model --exp-cfg config_files/smpl2smplh.yaml
  ```
This should be run from the top directory of the repository.

The file *smpl2smplh.yaml* contains a sample configuration that reads meshes from a folder,
processes them and returns pkl files with SMPL parameters. To run on your own data create a folder
with SMPL meshes, in either ply or obj format, change the path in the config file and run the code.

### SMPL+H to SMPL-X

To run the code to convert SMPL+H meshes to SMPL-X parameters use the following command:
  ```Shell
  python -m transfer_model --exp-cfg config_files/smplh2smplx.yaml
  ```
This should be run from the top directory of the repository.

The file *smplh2smplx.yaml* contains a sample configuration that reads meshes from a folder,
processes them and returns pkl files with SMPL-X parameters. To run on your own data create a folder
with SMPL+H meshes, in either ply or obj format, change the path in the config file and run the code.


### SMPL-X to SMPL+H

To run the code to convert SMPL-X meshes to SMPL+H parameters use the following command:
  ```Shell
  python -m transfer_model --exp-cfg config_files/smplx2smplh.yaml
  ```
This should be run from the top directory of the repository.

The file *smplx2smpl.yaml* contains a sample configuration that reads meshes from a folder,
processes them and returns pkl files with SMPL+H parameters. To run on your own data create a folder
with SMPL-X meshes, in either ply or obj format, change the path in the config file and run the code.
Make sure that you do not use the jaw pose and expression parameters to generate
the meshes.


## Visualize correspondences

To visualize correspondences:
```Shell
python vis_correspondences.py --exp-cfg configs/smpl2smplx.yaml --exp-opts colors_path PATH_TO_SMPL_COLORS
```
You should then see the following image. Points with similar color are in
correspondence.
![Correspondence example](./docs/images/smpl_smplx_correspondence.png)

## Citation

Depending on which model is loaded for your project, i.e. SMPL-X or SMPL+H or SMPL, please cite the most relevant work:

```
@article{SMPL:2015,
    author = {Loper, Matthew and Mahmood, Naureen and Romero, Javier and Pons-Moll, Gerard and Black, Michael J.},
    title = {{SMPL}: A Skinned Multi-Person Linear Model},
    journal = {ACM Transactions on Graphics, (Proc. SIGGRAPH Asia)},
    month = oct,
    number = {6},
    pages = {248:1--248:16},
    publisher = {ACM},
    volume = {34},
    year = {2015}
}
```

```
@article{MANO:SIGGRAPHASIA:2017,
          title = {Embodied Hands: Modeling and Capturing Hands and Bodies Together},
          author = {Romero, Javier and Tzionas, Dimitrios and Black, Michael J.},
          journal = {ACM Transactions on Graphics, (Proc. SIGGRAPH Asia)},
          volume = {36},
          number = {6},
          pages = {245:1--245:17},
          series = {245:1--245:17},
          publisher = {ACM},
          month = nov,
          year = {2017},
          url = {http://doi.acm.org/10.1145/3130800.3130883},
          month_numeric = {11}
        }
```


```
@inproceedings{SMPL-X:2019,
    title = {Expressive Body Capture: 3D Hands, Face, and Body from a Single Image},
    author = {Pavlakos, Georgios and Choutas, Vasileios and Ghorbani, Nima and Bolkart, Timo and Osman, Ahmed A. A. and Tzionas, Dimitrios and Black, Michael J.},
    booktitle = {Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
    year = {2019}
}
```


## Acknowledgments
The code of this repository was implemented by [Vassilis Choutas](vassilis.choutas@tuebingen.mpg.de),
based on a Chumpy implementation from [Timo Bolkart](timo.bolkart@tuebingen.mpg.de).

## Contact

For questions, please contact [smplx@tue.mpg.de](smplx@tue.mpg.de).
