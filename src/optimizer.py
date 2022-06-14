import torch

from torch.optim.lr_scheduler import OneCycleLR

def get_optimizer(self, args):
    args = args.copy()
    opt_type = args.pop("type")
    if opt_type == "adam":
        return get_adam(self, args)
    elif opt_type == "sgd":
        return get_sgd(self, args)

def get_adam(self, args):
    learning_rate = args.learning_rate
    num_epochs = args.num_epochs
    weight_decay = args.weight_decay

    optimizer = torch.optim.AdamW(
        self.model.parameters(), 
        lr = learning_rate, 
        weight_decay = weight_decay)
        
        
    scheduler = {
        'scheduler': OneCycleLR(
            optimizer, 
            max_lr = learning_rate, 
            steps_per_epoch = int(len(self.train_dataloader())), 
            epochs = num_epochs, 
            anneal_strategy = "linear", 
            final_div_factor = 30,), 
        'name': 'learning_rate', 
        'interval':'step', 
        'frequency': 1
    }

    return optimizer, scheduler

def get_sgd(self, args):
    learning_rate = args.learning_rate
    num_epochs = args.num_epochs
    weight_decay = args.weight_decay
    step_size = args.step_size
    gamma = args.gamma

    optimizer = torch.optim.SGD(
        self.model.parameters(), 
        lr = learning_rate, 
        weight_decay = weight_decay)
        
        
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size,
        gamma
    )

    return optimizer, scheduler
