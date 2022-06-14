from omegaconf import OmegaConf

def get_metric(args):
    if isinstance(args, str):
        return eval(args)
    elif OmegaConf.is_list(args.type):
        return merge([get_metric(_) for _ in args.type])
    else:
        return eval(args.type)

def merge(metric_funcs):
    def func(y, yhat):
        metrics = {}
        for f in metric_funcs:
            metric = f(y, yhat)
            metrics.update(metric)
        return metrics
    return func


def binary_accuracy(y, yhat):
    y = y.cpu().numpy()
    yhat = yhat.argmax(1).cpu().numpy()

    acc1 = (y[:,0] == yhat[:,0]).mean()
    acc2 = (y[:,1:][y[:,1:] != -1] == yhat[:,1:][y[:,1:] != -1]).mean()
    acc = (acc1 + acc2) / 2

    metrics = {"valid_metric": acc, "valid_acc_text": acc1, "valid_acc_tag": acc2}
    return metrics

def class_accuracy(y, yhat):
    y = y.cpu().numpy()
    yhat = yhat.argmax(1).cpu().numpy()

    metrics = {f"valid_acc{i}": (y[:,i][y[:,i] != -1] == yhat[:,i][y[:,i] != -1]).mean() for i in range(y.shape[1])}
    return metrics
