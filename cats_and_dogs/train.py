import time
from collections.abc import Callable

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from cats_and_dogs.constants import default_device


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int,
    criterion: Callable | None = None,
    opt: Optimizer | None = None,
    device: torch.device | None = None,
):
    criterion = criterion or nn.CrossEntropyLoss()
    opt = opt or torch.optim.Adam(model.parameters(), lr=1e-3)
    device = device or default_device()

    model.to(device)

    train_loss = []
    val_loss = []
    val_accuracy = []
    top_val_accuracy = -1
    best_model = None

    for epoch in tqdm(range(n_epochs), total=n_epochs, desc="Epochs"):
        ep_train_loss = []
        ep_val_loss = []
        ep_val_accuracy = []
        start_time = time.time()

        ### TRAINING PHASE
        model.train(True)  # enable dropout / batch_norm training behavior
        for x_batch, y_batch in tqdm(
            train_loader,
            total=len(train_loader),
            desc=f"Train batches (epoch {epoch + 1}/{n_epochs})",
            leave=False,
        ):
            # move data to target device
            # X_batch.cuda() - highly not recommended!
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            ## FORWARD PASS
            opt.zero_grad()
            # train on batch: compute loss, calc grads, perform optimizer step
            # and zero the grads
            predicts = model(x_batch)
            loss = criterion(predicts, y_batch)
            ## <\FORWARD PASS>

            ## BACKWARD PASS
            loss.backward()
            opt.step()
            ## </BACKWARD PASS>
            ep_train_loss.append(loss.item())

        ### VALIDATION PHASE
        model.train(False)  # disable dropout / use averages for batch_norm
        with torch.no_grad():  # alternatively inference_mode()
            for x_batch, y_batch in val_loader:
                # move data to target device
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)

                # compute predictions
                preds = model(x_batch)
                validation_loss = criterion(preds, y_batch)

                ep_val_loss.append(validation_loss.item())
                # y_pred = preds.max(1)[1].data
                preds_np = np.argmax(preds.detach().cpu().numpy(), 1).ravel()
                gt = y_batch.detach().cpu().numpy().ravel()
                hits = np.array(preds_np == gt)
                ep_val_accuracy.append(hits.astype(np.float32).mean())

        # print the results for this epoch:
        print(f"Epoch {epoch + 1} of {n_epochs} took {time.time() - start_time:.3f}s")

        train_loss.append(np.mean(ep_train_loss))
        val_loss.append(np.mean(ep_val_loss))
        val_accuracy.append(np.mean(ep_val_accuracy))

        print(f"\t  training loss: {train_loss[-1]:.6f}")
        print(f"\tvalidation loss: {val_loss[-1]:.6f}")
        print(f"\tvalidation accuracy: {100 * val_accuracy[-1]:.1f}")
        if val_accuracy[-1] > top_val_accuracy:
            best_model = model

    return train_loss, val_loss, val_accuracy, best_model


@torch.inference_mode()
def test_model(
    model,
    test_loader,
    subset="test",
    device: torch.device | None = None,
):
    device = device or default_device()

    model.train(False)  # disable dropout / use averages for batch_norm
    test_batch_acc = []

    for x_batch, y_batch in test_loader:
        data_device = x_batch.to(device)
        logits = model(data_device)
        y_pred = logits.max(1)[1].data
        test_batch_acc.append(np.mean((y_batch.cpu() == y_pred.cpu()).numpy()))

    test_accuracy = np.mean(test_batch_acc)

    print("Results:")
    print(f"    {subset} accuracy: {test_accuracy * 100:.2f} %")

    if test_accuracy > 0.9:
        print("  Amazing!")
    elif test_accuracy > 0.7:
        print("  Good!")
    else:
        print("  We need more magic! Follow instructions below")

    return test_accuracy
