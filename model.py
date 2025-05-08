import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import Sequential, Linear, Tanh, Embedding, FlattenConsecutive, Dropout


class DropoutModel:
    def __init__(self, words_file="names.txt", embedding_dim=27):
        self.words = open(words_file, "r").read().splitlines()
        chars = sorted(list(set("".join(self.words))))
        self.stoi = {s: i + 1 for i, s in enumerate(chars)}
        self.stoi["."] = 0
        self.itos = {i: s for s, i in self.stoi.items()}
        self.vocab_size = len(self.stoi)
        self.xs, self.ys = self.build_dataset()
        self.num = len(self.xs)
        self.n_hidden = 128
        self.n_embd = 10
        self.batch_size = 32
        self.num_epochs = 350000

    def build_dataset(self):
        xs, ys = [], []
        for w in self.words:
            chs = ["."] + list(w) + ["."]
            for ch1, ch2 in zip(chs, chs[1:]):
                ix1 = self.stoi[ch1]
                ix2 = self.stoi[ch2]
                xs.append(ix1)
                ys.append(ix2)
        return torch.tensor(xs), torch.tensor(ys)

    def train(self):
        model = Sequential(
            [
                Embedding(self.vocab_size, self.n_embd),
                Dropout(training_mode=True),
                Linear(self.n_embd, self.n_hidden, bias=False),
                Tanh(),
                Dropout(training_mode=True),
                Linear(self.n_hidden, self.n_hidden, bias=False),
                Tanh(),
                Dropout(training_mode=True),
                Linear(self.n_hidden, self.n_hidden, bias=False),
                Tanh(),
                Dropout(training_mode=True),
                Linear(self.n_hidden, self.vocab_size, scale=0.1),
            ]
        )
        self.model = model
        parameters = model.parameters()
        for p in parameters:
            p.requires_grad = True

        for k in range(self.num_epochs):
            ix = torch.randint(0, self.xs.shape[0], (self.batch_size,))
            Xb, Yb = self.xs[ix], self.ys[ix]  # batches

            # forward
            logits = model(Xb)
            loss = F.cross_entropy(logits, Yb)

            for p in parameters:
                p.grad = None  # set grad to 0
            loss.backward()

            # simple gradient descent
            lr = 0.1 if k < (self.num_epochs * 0.75) else 0.01
            for p in parameters:
                p.data += -lr * p.grad

            if k % 10000 == 0:
                print(f"Iteration {k}: loss = {loss.item():.4f}")

    #
    def generate_names(self, num_names=10):
        test_model = self.model
        for layer in test_model.layers:
            if isinstance(layer, Dropout):
                layer.training = False
        generated_names = []

        for _ in range(num_names):
            out = []
            context = [0]
            while True:
                x = torch.tensor([context])
                logits = self.model(x)
                logits = logits.squeeze(0)
                probs = F.softmax(logits, dim=1)

                ix = torch.multinomial(probs, num_samples=1).item()

                context = context[1:] + [ix]
                out.append(ix)

                if ix == 0:
                    break
            generated_names.append("".join(self.itos[i] for i in out))
        return generated_names
