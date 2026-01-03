**SPH-Net Price Prediction**

SPH-Net (Stock Price Hybrid Network) is a PyTorch implementation of a hybrid vision-transformer + transformer architecture for time-series forecasting of stock prices. This repository currently contains the model implementation in `model.py` and placeholder folders for `data/` and `models/`.

**What This Repository Contains**
- **`model.py`**: The full PyTorch implementation of the SPH-Net model (ViT-based feature extractor + Transformer predictor) and a `test_sphnet()` entrypoint demonstrating instantiation and a dummy forward pass.
- **`training.py`**: Placeholder for training scripts (currently empty). Implement your training loop here.
- **`data/`**: Data directory (currently empty). Place preprocessed datasets or CSVs here.
- **`models/`**: Checkpoint / export directory (currently empty). Saved PyTorch models go here.
- **`architechture.png`**: Model architecture image (visual reference).

**Requirements**
- Python 3.8+ (3.10+/3.14 works too)
- PyTorch (CPU or CUDA build)

Install the basics with pip:

```
pip install torch
```

If you need a specific CUDA-enabled wheel, follow PyTorch's official install instructions at https://pytorch.org.

**Quick Start**

1. Run the model test to verify the code and inspect the architecture:

```
python3 model.py
```

This runs `test_sphnet()` which prints the model summary, parameter counts, and performs a dummy forward pass. It also prints an example optimizer and loss setup.

2. Add your data under `data/` and implement training steps in `training.py`.

Example minimal training loop (to be placed in `training.py`):

```
from model import SPHNet
import torch

model = SPHNet()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()

# prepare your DataLoader yielding (inputs, targets)
for epoch in range(epochs):
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # save checkpoints to `models/`
    torch.save(model.state_dict(), f"models/sphnet_epoch_{epoch}.pt")
```

**Project Structure**
- `model.py` — model definitions and example `test_sphnet()` runnable script.
- `training.py` — (empty) intended location for training script and experiments.
- `data/` — place raw or preprocessed datasets here (CSV, NPZ, Torch tensors).
- `models/` — output directory for saved checkpoints and exported models.

**Notes & Next Steps**
- `training.py` is currently empty. If you want, I can scaffold a full training script, including dataset loader, preprocessing, logging, checkpointing, and argument parsing.
- Add README badges, tests, or a `requirements.txt` if you'd like reproducible environment pins.

**License**
This repo includes a `LICENSE -> apache 2.0` file at the project root.

---
If you'd like, I can now:
- scaffold a complete `training.py` with CLI args and DataLoader support,
- add a `requirements.txt` or `pyproject.toml`, or
- implement basic data preprocessing for common CSV price data.
