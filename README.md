## Solution for [GAIIC-Track1](https://www.heywhale.com/home/competition/620b34c41f3cf500170bd6ca)

### Introduction

Here is a basic part of codes for [GAIIC-Track1](https://www.heywhale.com/home/competition/620b34c41f3cf500170bd6ca). Other parts will be released after final.

### Data Structure

```sh
├── config.yaml
├── data
│   ├── preliminary_testA.txt
│   ├── preliminary_testB.txt
│   └── train
│       ├── attr_to_attrvals.json
│       ├── train_coarse.txt
│       └── train_fine.txt
├── README.md
├── Solver.py
├── Inferencer.py
├── data.ipynb
└── src
    ├── dataset.py
    ├── __init__.py
    ├── metrics.py
    ├── model.py
    └── optimizer.py
```

### Solution Idea

1. Visual-Bert was used to integrated multi-modality infomation, followed by a multi head MLP classifier for 13 binary classification task

2. Multiple augmentations were designed for robust training, including `        `

3. Model ensembling gave a good result

### Run

1. Run `data.ipynb` cell by cell, and make sure you specify `"preliminary_test(A or B).txt"` correctly for test set of different rounds.

2. Write your own configs in `config.yaml`:

    - **NOTE** 
    
    - The augmentation functions (`./src/dataset.py: JointDataset.aug(...)`) are not shown in codes, but replaced by place holders in comments.

    - Normal augmentations like replacing, removing, and shuffling can get a comparable score.

    - Multiple models ensemble is the key to get a better score.

    - Hyperarameters in `config.yaml` is the same as my final solution, except those augmentation args.

3. Run `python Solver.py --config config.yaml` to train models

    - GPU ids are specified at the first line of `Solver.py`, and multi-GPU training is supported (e.g. `gpus = "0,1"`)

4. Run `python Inferencer.py` to inference on test set. Before that, you should:

    - Set model checkpoints paths in `Inferencer.py`. `group_id` isn't necessary
    
    - Specify test set path of `X_test = np.load("./data/test(A or B).npy")` and `df_test = pd.read_csv("./data/test(A or B)_text.csv")` in `./src/dataset.py` 

### Results

Rounds|Place|Score
--|--|--
Preliminary-A|1st|0.95823330
Preliminary-B|1st|0.95555683
Semi-final-A|2nd|0.93149024
Semi-final-B|2nd|0.95485400
