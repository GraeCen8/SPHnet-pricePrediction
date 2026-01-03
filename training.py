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
    model, optimizer, trainLoader, valLoader, testLoader = accelerator.prepare(
        model, optimizer, trainLoader, valLoader, testLoader
    )

    #main loop
    for epoch in range(1,epochs):


        pbar = tqdm.tqdm(trainLoader, desc=f"TrainingEpoch 1 /{epochs}", unit="batch")

        #training section
        model.train()
        for batch in trainLoader:
            outputs = model(**batch)
            loss = criterion(outputs.logits, batch['labels'])
            acc.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            if scheduler:
                scheduler.step()
            pbar.update(1)
        pbar.set_description(f"in Epoch {epoch} /{epochs} | Train Loss: {loss.item():.4f}")
        pbar.refresh()

        #validation section
        if epoch % valGAP == 0:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in valLoader:
                    outputs = model(**batch)
                    loss = outputs.loss
                    val_losses.append(loss.item())
            avg_val_loss = np.mean(val_losses)
            print(f"Epoch {epoch}, Validation Loss: {avg_val_loss}")
            print(f"finished Epoch {epoch} /{epochs} | Val Loss: {avg_val_loss:.4f}")
            pbar.refresh()
        pbar.close()
    
    #testing section
    model.eval()
    test_losses = []
    with torch.no_grad():
        for batch in testLoader:
            outputs = model(**batch)
            loss = outputs.loss
            test_losses.append(loss.item())
    avg_test_loss = np.mean(test_losses)
    print(f"Testing Loss: {avg_test_loss}")