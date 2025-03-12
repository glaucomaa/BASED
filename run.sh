#!/bin/bash
#
# This script runs the main training script with the following parameters:
#
# --epochs       : Number of training epochs. (default: 8)
# --batch_size   : Batch size for training. (default: 16)
# --embed_dim    : Dimension of the embeddings. (default: 128)
# --num_layers   : Number of transformer layers. (default: 2)
# --num_heads    : Number of attention heads. (default: 4)
# --block_size   : Sequence block size in tokens. (default: 32)
# --window_size  : Sliding window size for local attention. (default: 8)
# --lr           : Learning rate. (default: 1e-3)
# --dropout      : Dropout rate used during training. (default: 0.1)
# --save_path    : File path to save the model weights. (default: model.pth)
# --progress     : Display a progress bar during training/evaluation. (default: False)
# --use_gpu      : To run on GPU. (default: False)
#
python main.py --epochs 1 --batch_size 16 --embed_dim 128 --num_layers 2 --num_heads 4 --block_size 32 --window_size 8 --lr 1e-3 --dropout 0.1 --save_path model.pth --progress
