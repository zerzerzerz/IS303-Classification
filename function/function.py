from ast import arg
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from dataset.dataset import MyDataset
from tqdm import tqdm
from os import makedirs
from os.path import join, isdir
from utils.utils import save_json, Logger
from sklearn.metrics import classification_report


def normalize(tensor):
    """
    Normalize to [-1, 1]
    """
    return ((tensor) / 255 - 0.5) * 2

def get_loss(input, target):
    """
    input.shape = B * C, where C is the number of class
    target.shape = B, where 0 <= target value < C
    """
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    return loss_fn(input, target.to(dtype=torch.int64))

def get_statistics_count(label, label_pred):
    """
    @label: ground truth of label, torch.IntTensor, shape = (B,)
    @label_pred: prediction of label, torch.IntTensor, shape = (B,)
    @return:
        TP, FP, TN, FN
    """
    tp = fp = tn = fn = 0
    for label1, label2 in zip(label, label_pred):
        label1, label2 = label1.item(), label2.item()
        if label1 == 0 and label2 == 0:
            tn += 1
        elif label1 == 0 and label2 != 0:
            fp += 1
        elif label1 != 0 and label2 == 0:
            fn += 1
        elif label1 != 0 and label2 != 0:
            tp += 1
        else:
            raise RuntimeError(f"label_gt={label1}, label2={label2}")
    return tp, fp, tn, fn



def run(
    model,
    args,
):
    
    
    optimizer = None if args.only_eval else Adam(model.parameters(), lr=args.lr)

    dirs = []
    base_dir = args.output_dir
    dirs.append(base_dir)
    checkpoints_dir = join(base_dir,'checkpoints')
    dirs.append(checkpoints_dir)
    for d in dirs:
        if not isdir(d):
            makedirs(d)

    save_json(vars(args), join(base_dir, 'args.json'))
    train_logger = None if args.only_eval else Logger(join(base_dir,'train.txt'))
    test_logger = Logger(join(base_dir,'test.txt'))

    # set device
    test_logger.log("Loading model...")
    device = torch.device(args.device)
    model.to(device)

    # load data
    if not args.only_eval:
        train_logger.log("Loading train dataset...")
        train_dataset = MyDataset(args.train_file_json_path,data_dim=min(args.data_dim, args.MAX_DATA_DIM))
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_logger.log("Loading test dataset")
    test_dataset = MyDataset(args.test_file_json_path, data_dim=min(args.data_dim, args.MAX_DATA_DIM))
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
        
    for epoch in tqdm(range(args.num_epoch)):
        if not args.only_eval:
            with torch.set_grad_enabled(True):
                model.train()
                loss_train = 0.0
                acc_train = 0.0
                TP = FP = TN = FN = 0
                count = 0
                for item in tqdm(train_dataloader):
                    img = normalize(item[0].to(device))
                    label = item[1].to(device).squeeze()
                    label_pred_distribution = model(img)
                    label_pred = label_pred_distribution.argmax(dim=1,keepdim=False).squeeze().to(dtype=torch.int32)
                acc_train += (label == label_pred).sum().item()
                tp, fp, tn, fn = get_statistics_count(label, label_pred)
                TP += tp
                FP += fp
                TN += tn
                FN += fn
                count += label.shape[0]

                l = get_loss(label_pred_distribution, label)
                loss_train += l.sum().item()
                l = l.mean()
                l.backward()
                optimizer.step()
                optimizer.zero_grad()

            acc_train /= count
            loss_train /= count
            TPR = TP / (TP + FN)
            FPR = FP / (TN + FP)
            TNR = TN / (TN + FP)
            FNR = FN / (TP + FN)
            train_logger.log("Epoch: %d/%d, acc = %.4f, loss = %.4f, TPR = %.4f, FPR = %.4f, TNR = %.4f FNR = %.4f" % (epoch, args.num_epoch, acc_train, loss_train, TPR, FPR, TNR, FNR))
            if((epoch + 1) % args.gap_epoch_save_checkpoint == 0):
                model.cpu()
                torch.save(model,join(checkpoints_dir,'epoch_%04d.pt'%epoch))
                model.to(device=device)
        
        with torch.set_grad_enabled(False):
            model.eval()
            loss_test = 0.0
            acc_test = 0.0
            acc_best = 0.0
            TP = FP = TN = FN = 0
            count = 0
            for item in tqdm(test_dataloader):
                img = normalize(item[0].to(device))
                label = item[1].to(device).squeeze()
                label_pred_distribution = model(img)

                label_pred = label_pred_distribution.argmax(dim=1,keepdim=False).squeeze().to(dtype=torch.int32)
                acc_test += (label == label_pred).sum().item()
                count += label.shape[0]
                tp, fp, tn, fn = get_statistics_count(label, label_pred)
                TP += tp
                FP += fp
                TN += tn
                FN += fn
                l = get_loss(label_pred_distribution, label)
                loss_test += l.sum().item()


            acc_test /= count
            loss_test /= count
            TPR = TP / (TP + FN)
            FPR = FP / (TN + FP)
            TNR = TN / (TN + FP)
            FNR = FN / (TP + FN)
            test_logger.log("Epoch: %d/%d, acc = %.4f, loss = %.4f, TPR = %.4f, FPR = %.4f, TNR = %.4f FNR = %.4f" % (epoch, args.num_epoch, acc_test, loss_test, TPR, FPR, TNR, FNR))
            if acc_test > acc_best:
                acc_best = acc_test
                model.cpu()
                torch.save(model,join(checkpoints_dir,'best_acc.pt'))
                model.to(device=device)

    model.cpu()
    torch.save(model,join(checkpoints_dir,'epoch_final.pt'))



def get_report(
    model,
    args,
):
    

    base_dir = args.output_dir
    device = torch.device(args.device)
    model.to(device)

    test_dataset = MyDataset(args.test_file_json_path, data_dim=min(args.data_dim, args.MAX_DATA_DIM))
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
        
    with torch.set_grad_enabled(False):
        model.eval()
        gts = []
        preds = []
        for item in tqdm(test_dataloader):
            img = normalize(item[0].to(device))
            label = item[1].to(device).squeeze()
            label_pred_distribution = model(img)

            label_pred = label_pred_distribution.argmax(dim=1,keepdim=False).squeeze().to(dtype=torch.int32)
            preds.append(label_pred)
            gts.append(label)
        gts = torch.concat(gts,dim=0).detach().cpu().numpy()
        preds = torch.concat(preds,dim=0).detach().cpu().numpy()
        report = classification_report(gts,preds,target_names=[f"Class_{c}" for c in range(10)])
        with open(join(base_dir,'report.txt'),'w') as f:
            f.write(report)
        return report
