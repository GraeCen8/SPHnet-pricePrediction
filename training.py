import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tqdm 
import accelerate as accelerator

def Train(model, optimizer,
            trainLoader, valLoader,
            testLoader, scheduler,
            metrics, epochs,
            valGAP = 2, criterion = None
                ):
    
        #set up accelerator
    acc = accelerator.Accelerator()
    model, optimizer, trainLoader, valLoader, testLoader = acc.prepare(
        model, optimizer, trainLoader, valLoader, testLoader
    )

    #main loop
    for epoch in range(1, epochs + 1):

        pbar = tqdm.tqdm(trainLoader, desc=f"TrainingEpoch {epoch} /{epochs}", unit="batch")

        # training section
        model.train()
        train_losses = []
        for batch in trainLoader:
            # dataloader yields (inputs, targets)
            inputs, targets = batch

            # Ensure targets have same shape as model output (B, 1)
            if targets.dim() == 1:
                targets = targets.unsqueeze(-1)

            # zero gradients (try set_to_none for efficiency, fallback if not supported)
            try:
                optimizer.zero_grad(set_to_none=True)
            except TypeError:
                try:
                    optimizer.zero_grad()
                except Exception:
                    pass

            outputs = model(inputs)

            loss = criterion(outputs, targets)
            acc.backward(loss)

            optimizer.step()
            if scheduler:
                scheduler.step()

            train_losses.append(loss.item())
            pbar.update(1)

        avg_train_loss = np.mean(train_losses) if train_losses else 0.0
        pbar.set_description(f"Epoch {epoch} /{epochs} | Train Loss: {avg_train_loss:.4f}")
        pbar.refresh()

        # validation section
        if epoch % valGAP == 0:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in valLoader:
                    inputs, targets = batch
                    if targets.dim() == 1:
                        targets = targets.unsqueeze(-1)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_losses.append(loss.item())
            avg_val_loss = np.mean(val_losses) if val_losses else 0.0
            print(f"Epoch {epoch}, Validation Loss: {avg_val_loss:.6f}")
            pbar.refresh()
        pbar.close()
    
    #testing section
    model.eval()
    test_losses = []
    with torch.no_grad():
        for batch in testLoader:
            inputs, targets = batch
            if targets.dim() == 1:
                targets = targets.unsqueeze(-1)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_losses.append(loss.item())
    avg_test_loss = np.mean(test_losses)
    print(f"Testing Loss: {avg_test_loss}")
    # return the unwrapped model so callers can save or inspect it
    try:
        final_model = acc.unwrap_model(model)
    except Exception:
        final_model = model

    return final_model