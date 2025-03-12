#!/usr/bin/env python
import argparse
import math
import os

import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import BasedTransformer, TextDataset
from utils import build_vocab, get_cosine_schedule_with_warmup


def train_epoch(model, dataloader, optimizer, scheduler, device, use_progress=False):
    model.train()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    if use_progress:
        data_iter = tqdm(enumerate(dataloader), total=len(dataloader), desc="Training")
    else:
        data_iter = enumerate(dataloader)

    for step, batch in data_iter:
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

        if step % 200 == 0 and not use_progress:
            current_lr = scheduler.get_last_lr()[0]
            print(f"Step {step}, Loss: {loss.item():.4f}, LR: {current_lr:.6f}")

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device, use_progress=False):
    model.eval()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    if use_progress:
        data_iter = tqdm(dataloader, desc="Evaluating")
    else:
        data_iter = dataloader

    with torch.no_grad():
        for batch in data_iter:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity


def main(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.use_gpu else "cpu"
    )
    print("Using device:", device)

    print("Loading dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    train_texts = dataset["train"]["text"]
    val_texts = dataset["validation"]["text"]

    print("Building vocabulary...")
    vocab = build_vocab(train_texts, min_freq=2)
    vocab_size = len(vocab)
    print(f"Vocab size: {vocab_size}")

    block_size = args.block_size
    train_dataset = TextDataset(train_texts, vocab, block_size=block_size)
    val_dataset = TextDataset(val_texts, vocab, block_size=block_size)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    model = BasedTransformer(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        block_size=block_size,
        window_size=args.window_size,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_training_steps = len(train_loader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=100, num_training_steps=total_training_steps
    )

    if args.load_path and os.path.exists(args.load_path):
        print(f"Loading model weights from {args.load_path} ...")
        model.load_state_dict(torch.load(args.load_path, map_location=device))

    print("Starting training...")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            use_progress=args.progress,
        )
        val_loss, val_ppl = evaluate(
            model, val_loader, device, use_progress=args.progress
        )
        print(
            f"Epoch {epoch}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Perplexity: {val_ppl:.2f}"
        )

    if args.save_path:
        torch.save(model.state_dict(), args.save_path)
        print(f"Model saved to {args.save_path}")

    print("Loading saved weights to verify...")
    new_model = BasedTransformer(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        block_size=block_size,
        window_size=args.window_size,
        dropout=args.dropout,
    ).to(device)
    new_model.load_state_dict(torch.load(args.save_path, map_location=device))
    _, new_ppl = evaluate(new_model, val_loader, device)
    print(f"Loaded model perplexity: {new_ppl:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Proof-of-concept Based Transformer with Regularization & LR Scheduling"
    )
    parser.add_argument(
        "--epochs", type=int, default=8, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--embed_dim", type=int, default=128, help="Embedding dimension"
    )
    parser.add_argument(
        "--num_layers", type=int, default=2, help="Number of transformer layers"
    )
    parser.add_argument(
        "--num_heads", type=int, default=4, help="Number of attention heads"
    )
    parser.add_argument(
        "--mlp_ratio", type=float, default=4.0, help="MLP expansion ratio"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--block_size", type=int, default=32, help="Sequence block size (tokens)"
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=8,
        help="Sliding window size for local attention",
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument(
        "--save_path", type=str, default="model.pth", help="Path to save model weights"
    )
    parser.add_argument(
        "--load_path",
        type=str,
        default="",
        help="Path to load model weights (optional)",
    )
    parser.add_argument(
        "--use_gpu", action="store_true", help="Flag to use GPU if available"
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Display progress bar during training/evaluation",
    )
    args = parser.parse_args()

    main(args)
