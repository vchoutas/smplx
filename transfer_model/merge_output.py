# merges the output of the main transfer_model script

import torch
from pathlib import Path
import pickle
from scipy.spatial.transform import Rotation as R

KEYS = [
"transl",
"global_orient",
"body_pose",
"betas",
"left_hand_pose",
"right_hand_pose",
"jaw_pose",
"leye_pose",
"reye_pose",
"expression",
"vertices",
"joints",
"full_pose",
"v_shaped",
"faces"
]

IGNORED_KEYS = [
"vertices",
"faces",
"v_shaped"
]

def aggregate_rotmats(x):
    x = torch.cat(x, dim=0).detach().numpy()
    s = x.shape[:-2]
    x = R.from_matrix(x.reshape(-1, 3, 3)).as_rotvec()
    x = x.reshape(s[0], -1)
    return x

aggregate_function = {k: lambda x: torch.cat(x, 0).detach().numpy() for k in KEYS}
aggregate_function["betas"] = lambda x: torch.cat(x, 0).mean(0).detach().numpy()

for k in ["global_orient", "body_pose", "left_hand_pose", "right_hand_pose", "jaw_pose", "full_pose"]:
    aggregate_function[k] = aggregate_rotmats

def merge(output_dir, gender):
    output_dir = Path(output_dir)
    assert output_dir.exists()
    assert output_dir.is_dir()

    # get list of all pkl files in output_dir with fixed length numeral names
    pkl_files = [f for f in output_dir.glob("*.pkl") if f.stem != "merged"]
    pkl_files = [f for f in sorted(pkl_files, key=lambda x: int(x.stem))]
    assert "merged.pkl" not in [f.name for f in pkl_files]

    merged = {}
    # iterate over keys and put all values in lists
    keys = set(KEYS) - set(IGNORED_KEYS)
    for k in keys:
        merged[k] = []
    for pkl_file in pkl_files:
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)
        for k in keys:
            if k in data:
                merged[k].append(data[k])
    b = torch.cat(merged["betas"], 0)
    print("betas:")
    for mu, sigma in zip(b.mean(0), b.std(0)):
        print("  {:.3f} +/- {:.3f}".format(mu, sigma))

    # aggregate all values
    for k in keys:
        merged[k] = aggregate_function[k](merged[k])

    # add gender
    merged["gender"] = gender

    # save merged data to same output_dir
    with open(output_dir / "merged.pkl", "wb") as f:
        pickle.dump(merged, f)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Merge output of transfer_model script')
    parser.add_argument('output_dir', type=str, help='output directory of transfer_model script')
    parser.add_argument('--gender', type=str, choices=['male', 'female', 'neutral'], help='gender of actor in motion sequence')
    args = parser.parse_args()
    merge(args.output_dir, args.gender)
