import json
import numpy as np
import pandas as pd
from Solver import *

models = [
    # >>>>
    # position holder for models
    # item format: (group_id, model checkpoint path)
    # e.g. models = [
    #   (1, './path/to/model1.ckpt'),
    #   (1, './path/to/model2.ckpt'),
    #   (2, './path/to/model3.ckpt'),
    #   (2, './path/to/model4.ckpt'),
    # ]
    # <<<<
]

trainer = pl.Trainer(gpus=len(gpus.split(",")), logger = False)
preds_list = []
for group_id, model in models:
    model = Model.load_from_checkpoint(model)
    outputs = trainer.predict(model)
    logits = torch.cat([_ for _ in outputs])
    preds = logits.softmax(1)[:,1].cpu().numpy()
    preds_list.append((group_id, preds))

preds_all = 0
for _, preds in preds_list:
    preds_all += preds
preds_all /= len(preds_list)
    

preds = preds_all
thres = np.array([0.6] + [0.5] * 12)

name2id = pd.read_csv("./data/name_to_id_B.csv")

with open("submissions/B/1.txt", "w") as f:
    for idx, (img_name, img_idx) in name2id.iterrows():
        pred, mask = preds[img_idx], np.array(model.ds_test.masks[img_idx])
        keys = model.ds_test.keys[img_idx]
        pred = (pred > thres).astype(int)
        pred = pred[mask == 1]
        row = {}
        row["img_name"] = img_name
        row["match"] = {
            "图文": int(pred[0]),
            **{k: int(v) for k, v in zip(keys, pred[1:])}
        }
        f.write(json.dumps(row, ensure_ascii = False) + "\n")