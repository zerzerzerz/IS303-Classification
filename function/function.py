import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from dataset.dataset import MyDataset
from tqdm import tqdm
from os import makedirs
from os.path import join, isdir
from utils.utils import save_json, Logger


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



def run(
    model,
    args,
):
    # set device
    device = torch.device(args.device)
    model.to(device)

    # load data
    train_dataset = MyDataset(args.train_file_json_path,data_dim=min(args.data_dim, args.MAX_DATA_DIM))
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_dataset = MyDataset(args.test_file_json_path, data_dim=min(args.data_dim, args.MAX_DATA_DIM))
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    
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
        
    for epoch in tqdm(range(args.num_epoch)):
        if not args.only_eval:
            with torch.set_grad_enabled(True):
                model.train()
                loss_train = 0.0
                acc_train = 0.0
                count = 0
                for item in tqdm(train_dataloader):
                    img = normalize(item[0].to(device))
                    label = item[1].to(device).squeeze()
                    label_pred_distribution = model(img)

                    label_pred = label_pred_distribution.argmax(dim=1,keepdim=False).squeeze().to(dtype=torch.int32)
                acc_train += (label == label_pred).sum().item()
                count += label.shape[0]

                l = get_loss(label_pred_distribution, label)
                loss_train += l.sum().item()
                l = l.mean()
                l.backward()
                optimizer.step()
                optimizer.zero_grad()

            acc_train /= count
            loss_train /= count
            train_logger.log("Epoch: %d/%d, acc = %.4f, loss = %.4f" % (epoch, args.num_epoch, acc_train, loss_train))
            if((epoch + 1) % args.gap_epoch_save_checkpoint == 0):
                model.cpu()
                torch.save(model,join(checkpoints_dir,'epoch_%04d.pt'%epoch))
                model.to(device=device)
        
        with torch.set_grad_enabled(False):
            model.eval()
            loss_test = 0.0
            acc_test = 0.0
            acc_best = 0.0
            count = 0
            for item in tqdm(test_dataloader):
                img = normalize(item[0].to(device))
                label = item[1].to(device).squeeze()
                label_pred_distribution = model(img)

                label_pred = label_pred_distribution.argmax(dim=1,keepdim=False).squeeze().to(dtype=torch.int32)
                acc_test += (label == label_pred).sum().item()
                count += label.shape[0]

                l = get_loss(label_pred_distribution, label)
                loss_test += l.sum().item()


            acc_test /= count
            loss_test /= count
            test_logger.log("Epoch: %d/%d, acc = %.4f, loss = %.4f" % (epoch, args.num_epoch, acc_test, loss_test))
            if acc_test > acc_best:
                acc_best = acc_test
                model.cpu()
                torch.save(model,join(checkpoints_dir,'best_acc.pt'))
                model.to(device=device)

    model.cpu()
    torch.save(model,join(checkpoints_dir,'epoch_final.pt'))