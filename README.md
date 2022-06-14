# Introduction

Here is a basic part of codes for [GAIIC-Track1](https://www.heywhale.com/home/competition/620b34c41f3cf500170bd6ca). Other parts will be released after final.

# Data Structure

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

# Run

1. Run `data.ipynb` cell by cell, and make sure you specify `"preliminary_test(A or B).txt"` correctly for test set of different rounds.

2. Write your own config in `config.yaml`

3. Run `python Solver.py --config config.yaml` to train models

4. Set model paths in `Inferencer.py`, and specify test set path in `./src/dataset.py`, and run `python Inferencer.py` to inference on test set.