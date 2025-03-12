import math

from torch.optim.lr_scheduler import LambdaLR


def build_vocab(texts, min_freq=1):
    freq = {}
    for text in texts:
        for token in text.split():
            freq[token] = freq.get(token, 0) + 1
    tokens = [t for t, count in freq.items() if count >= min_freq]
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for t in tokens:
        vocab[t] = len(vocab)
    return vocab


def tokenize(text, vocab):
    return [vocab.get(token, vocab["<UNK>"]) for token in text.split()]


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / max(1, num_warmup_steps)
        return 0.5 * (
            1
            + math.cos(
                math.pi
                * (current_step - num_warmup_steps)
                / (num_training_steps - num_warmup_steps)
            )
        )

    return LambdaLR(optimizer, lr_lambda)
