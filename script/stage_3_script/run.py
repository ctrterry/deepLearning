# File: script/stage_3_script/run.py
#!/usr/bin/env python3
import os
import argparse
import torch
from local_code.stage_3_code.stage3 import main_stage3

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stage3 CNN Training')
    parser.add_argument('--data-dir',      type=str,
        default=os.path.abspath(os.path.join(os.path.dirname(__file__),
                                            '../../data/stage_3_data')))
    parser.add_argument('--result-dir',    type=str,
        default=os.path.abspath(os.path.join(os.getcwd(),
                                            'result/stage_3_result')))
    parser.add_argument('--epochs',        type=int,   default=10,
        help='Epochs for MNIST & ORL')
    parser.add_argument('--epochs-cifar',  type=int,   default=None,
        help='Epochs for CIFAR-10 (overrides --epochs if set)')
    parser.add_argument('--lr',            type=float, default=1e-3)
    parser.add_argument('--weight-decay',  type=float, default=1e-4)
    parser.add_argument('--bs-mnist',      type=int,   default=64)
    parser.add_argument('--bs-face',       type=int,   default=32)
    parser.add_argument('--bs-cifar',      type=int,   default=64)
    parser.add_argument('--device',        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()
    print(f"Using device: {args.device}")
    os.makedirs(args.result_dir, exist_ok=True)

    main_stage3(
        data_dir     = args.data_dir,
        result_dir   = args.result_dir,
        device_str   = args.device,
        epochs       = args.epochs,
        epochs_cifar = args.epochs_cifar,
        lr           = args.lr,
        weight_decay = args.weight_decay,
        bs_mnist     = args.bs_mnist,
        bs_face      = args.bs_face,
        bs_cifar     = args.bs_cifar
    )
