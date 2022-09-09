## Removing Chumpy objects

In a Python 2 virtual environment with [Chumpy](https://github.com/mattloper/chumpy) installed run the following to remove any Chumpy objects from the model data:

```bash
python tools/clean_ch.py --input-models path-to-models/*.pkl --output-folder output-folder
```

## Merging SMPL-H and MANO parameters

In order to use the given PyTorch SMPL-H module we first need to merge the SMPL-H and MANO parameters in a single file. After agreeing to the license and downloading the models, run the following command:

```bash
python tools/merge_smplh_mano.py --smplh-fn SMPLH_FOLDER/SMPLH_GENDER.pkl \
 --mano-left-fn MANO_FOLDER/MANO_LEFT.pkl \
 --mano-right-fn MANO_FOLDER/MANO_RIGHT.pkl \
 --output-folder OUTPUT_FOLDER
```

where SMPLH_FOLDER is the folder with the SMPL-H files and MANO_FOLDER the one for the MANO files.


### SMPL-H version used in AMASS

For AMASS, you should download the body with 16 betas, here is the process:

```
- Download the zip folder from "Models & Code" and extract it to get the folder `mano_v1_2`
- Download the zip folder from "Extended SMPL+H model" and extract it to get the folder `smplh`

$ git clone https://github.com/vchoutas/smplx.git
$ cd smplx
$ python tools/merge_smplh_mano.py \
--smplh-fn /path/to/smplh/female/model.npz \
--mano-left-fn /path/to/mano_v1_2/models/MANO_LEFT.pkl \
--mano-right-fn /path/to/mano_v1_2/models/MANO_RIGHT.pkl \
--output-folder /path/to/smplh/merged

cp /path/to/smplh/merged/model.pkl /path/to/smplx_models/smplh/SMPLH_FEMALE.pkl
```

In the end you get the smplh model required by smplx 'smplx_models/smplh/SMPLH_FEMALE.pkl'
