import json
import numpy as np
import pandas as pd
import jieba
from sklearn.model_selection import KFold

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import BertTokenizer

def get_data(args):
    data_type = args.type
    fold = args.get("fold", -1)
    batch_size = args.batch_size
    tokenizer = BertTokenizer.from_pretrained(args.model_name)

    jieba.load_userdict("./data/train/jieba.dict")
    X = np.load("./data/train/train.npy")
    X_test = np.load("./data/testB.npy")
    df = pd.read_csv("./data/train/train_text.csv")
    df_test = pd.read_csv("./data/testB_text.csv")
    for k in ("keys", "values"):
        df[k] = df[k].apply(eval)
        df_test[k] = df_test[k].apply(eval)


    split = KFold(5, random_state = 0, shuffle = True)
    train_idx, valid_idx = list(split.split(X))[fold]
    if fold == -1:
        train_idx = np.concatenate([train_idx, valid_idx])

    if data_type == "JointData":
        DataClass = JointDataset
    else:
        raise NotImplementedError()

    
    train_args = {
        "df": df.loc[train_idx].reset_index(drop = True),
        "X": X,
        "tokenizer": tokenizer,
        "phase": "train",
        **args         
    }
    valid_args = {
        "df": df.loc[valid_idx].reset_index(drop = True),
        "X": X,
        "tokenizer": tokenizer,
        "phase": "val",
        **args         
    }
    test_args = {
        "df": df_test,
        "X": X_test,
        "tokenizer": tokenizer,
        "phase": "test",
        **args         
    }

    ds_train = DataClass(**train_args)
    ds_valid = DataClass(**valid_args)
    ds_test = DataClass(**test_args)

    def dl_train(shuffle = True, drop_last = True, num_workers = 4):
        return DataLoader(ds_train, batch_size, shuffle = shuffle, drop_last = drop_last, num_workers = num_workers)

    def dl_valid(shuffle = False, num_workers = 4):
        return DataLoader(ds_valid, batch_size, shuffle = shuffle, num_workers = num_workers)

    def dl_test(shuffle = False, num_workers = 4):
        return DataLoader(ds_test, batch_size, shuffle = shuffle, num_workers = num_workers)

    return (ds_train, ds_valid, ds_test), (dl_train, dl_valid, dl_test)

class Keys:
    with open("./data/train/attr_to_attrvals.json") as f:
        atts = json.load(f)
        keys = ['图文', '版型', '穿着方式', '类别', '衣长', '袖长', '裙长', '裤型', '裤长', '裤门襟', '闭合方式', '鞋帮高度', '领型']
        key_ids = {k:i for i, k in enumerate(keys)}

        for k in atts:
            att = {}
            for i, vi in enumerate(sorted(atts[k])):
                for vii in vi.split("="):
                    att[vii] = i
            atts[k] = att

    def sample(self, k, v):
        return np.random.choice([_ for _ in self.atts[k] if self.atts[k][_] != self.atts[k][v]])

    def __getitem__(self, x):
        if x in self.key_ids:
            return self.key_ids[x]
        else:
            return self.keys[x]

    def __len__(self):
        return len(self.keys)


class JointDataset(Dataset):
    def __init__(self, df, X, tokenizer, phase,
            max_length = 32,
            # >>>>
            # position holder for augmentation args
            # <<<<
            *args, **kwargs):
        self.X = X
        self.tokenizer = tokenizer
        self.phase = phase
        self.max_length = max_length

        
        # >>>>
        # position holder for augmentation args
        # <<<<

        self.K = Keys()



        self.texts, self.img_idxes = df['text'].tolist(), df['img_idx'].tolist()
        self.keys, self.values = df['keys'].tolist(), df['values'].tolist()
        self.it_labels = df['label'].tolist()
        self.labels, self.masks = self.gen_label()

        # >>>>
        # position holder for augmentation args
        # <<<<

    def gen_label(self):
        labels = []; masks = []
        for idx in range(len(self.texts)):
            label = [0] * len(self.K); mask = [0] * len(self.K)
            label[0] = self.it_labels[idx]; 
            mask[0] = 1
            for i, k in enumerate(self.keys[idx]):
                label[self.K[k]] = 1
                mask[self.K[k]] = 1
            labels.append(label); masks.append(mask)
        return labels, masks

    # >>>>
    # position holder for augmentation functions
    # <<<<

    def aug(self, text, keys, values, labels, masks):
        # >>>>
        # position holder for augmentation functions
        # <<<<

        return text, keys, values, labels, masks

    def __getitem__(self, idx):
        x = self.X[self.img_idxes[idx]].astype(np.float32)
        x = x.reshape(1, -1)
        x_token_type_ids = np.ones(x.shape[:-1], dtype = int)
        x_attention_mask = np.ones(x.shape[:-1], dtype = np.float32)

        keys, values = self.keys[idx].copy(), self.values[idx].copy()
        text, labels = self.texts[idx], self.labels[idx].copy()
        masks = self.masks[idx].copy()

        text, keys, values, labels, masks = self.aug(text, keys, values, labels, masks)
        
        tok = self.tokenizer.encode_plus(
            text, 
            max_length = self.max_length, 
            truncation = True, 
            padding = "max_length")
        tok = {k: np.array(v) for k, v in tok.items()}
        tok.update({
            "visual_embeds": x,
            "visual_token_type_ids": x_token_type_ids,
            "visual_attention_mask": x_attention_mask,
        })

        labels = np.array(labels)
        masks = np.array(masks)
        labels[masks != 1] = -1
        
        if self.phase == "test":
            return tok, labels
        else:
            return tok, labels

    def __len__(self):
        return len(self.texts)
