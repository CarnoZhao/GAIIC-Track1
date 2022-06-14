import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers


class MLP(nn.Module):
    def __init__(self, num_channels, drop_rate, act_type):
        super(MLP, self).__init__()
        if act_type == "relu":
            act = nn.ReLU
        elif act_type == "prelu":
            act = nn.PReLU
        elif act_type == "leakyrelu":
            act = nn.LeakyReLU

        fcs = [nn.Linear(num_channels[0], num_channels[1])]
        for i in range(1, len(num_channels) - 1):
            fcs.extend([
                nn.BatchNorm1d(num_channels[i]),
                act(),
                nn.Dropout(drop_rate),
                nn.Linear(num_channels[i], num_channels[i + 1])
            ])

        self.fcs = nn.Sequential(*fcs)

    def forward(self, x):
        out = self.fcs(x)
        return out

class MultiHeadClassifier(nn.Module):
    def __init__(self, num_heads, num_channel, num_classes, drop_rate=0.5, act_type="relu"):
        super(MultiHeadClassifier, self).__init__()
        heads = []
        nums_channels = [num_channel, num_channel, num_classes]
        for i in range(num_heads):
            heads.append(MLP(nums_channels, drop_rate, act_type))

        self.heads = nn.ModuleList(heads)

    def forward(self, x):
        logits = [_(x) for _ in self.heads]
        return torch.stack(logits, dim = -1)

class MultiHeadVisualBert(nn.Module):
    def __init__(self, args):
        super(MultiHeadVisualBert, self).__init__()
        model_name = args.model_name
        num_classes = args.num_classes
        num_heads = args.num_heads
        drop_rate = args.drop_rate
        act_type = args.get("act_type", "relu")
        self.use_pool = args.get("use_pool", True)

        ref_model = transformers.BertModel.from_pretrained(model_name)
        self.model = transformers.VisualBertModel(transformers.VisualBertConfig(
            num_hidden_layers = args.get("num_hidden_layers", ref_model.config.num_hidden_layers),
            hidden_size = ref_model.config.hidden_size,
            vocab_size = ref_model.config.vocab_size, 
            intermediate_size = ref_model.config.intermediate_size,
            num_attention_heads = ref_model.config.num_attention_heads,
            visual_embedding_dim = 2048,
            return_dict = True,
            output_hidden_states = True))
        self.model.load_state_dict(ref_model.state_dict(), strict = False)

        self.heads = MultiHeadClassifier(
            num_heads,
            self.model.config.hidden_size,
            num_classes,
            drop_rate,
            act_type,
        )

    def forward(self, x):
        out = self.model(**x)
        if self.use_pool:
            out = out[1]
        else:
            out = torch.stack(out.hidden_states, -1)[:,0,:,-4:].mean(-1)
        out = self.heads(out)
        return out

class MaskCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index):
        super(MaskCrossEntropyLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()
        self.ignore_index = ignore_index

    def forward(self, yhat, y):
        yhat = yhat.permute(0, 2, 1)
        return self.ce(yhat[y != self.ignore_index], y[y != self.ignore_index])

def get_model(args):
    return eval(args.type)(args)

def get_loss(args):
    args = args.copy()
    loss_type = args.pop("type")
    try:
        return eval(loss_type)(**args)
    except:
        return eval("nn." + loss_type)(**args)