from model.model import Model
from function.function import run
from utils.utils import setup_seed
import torch
import argparse
from os.path import isfile

if __name__ == "__main__":
    setup_seed()

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--gap_epoch_save_checkpoint", type=int, default=5, help="how often to save checkpoint")
    parser.add_argument("--data_dim", type=int, default=1024)
    parser.add_argument("--MAX_DATA_DIM", type=int, default=1024)
    parser.add_argument("--num_layer", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--train_file_json_path", type=str, default="data/train.json")
    parser.add_argument("--test_file_json_path", type=str, default="data/test.json")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="result/0502-debug-2")
    parser.add_argument("--only_eval", type=bool, default=False)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")


    args = parser.parse_args()
    args.data_dim = min(args.data_dim, args.MAX_DATA_DIM)
    args.num_epoch = 1 if args.only_eval else args.num_epoch
    if args.checkpoint_path is None or args.checkpoint_path == "":
        args.checkpoint_path = None
    else:
        args.checkpoint_path = args.checkpoint_path if isfile(args.checkpoint_path) else None
    
    model = Model(input_dim=args.data_dim, num_layers=args.num_layer) if args.checkpoint_path is None else torch.load(args.checkpoint_path)
    run(model, args)