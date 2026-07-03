"""Pretrain ZDT flow and diffusion models from .npy training data."""

import argparse
from pathlib import Path

import numpy as np
import torch
from diffusiongym.schedulers import DiffusionScheduler, OptimalTransportScheduler
from diffusiongym.types import DDTensor
from genexp.base_models.mlp import TensorMLPModel
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader, TensorDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain ZDT flow and diffusion models from .npy data.")
    parser.add_argument("--train_data_path", type=Path, required=True)
    parser.add_argument("--output_model_path", type=Path, required=True)
    parser.add_argument("--model_type", choices=("diffusion", "flow"), required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--beta_0", type=float, default=0.1)
    parser.add_argument("--beta_1", type=float, default=12.0)
    return parser.parse_args()


def load_training_data(train_data_path: Path) -> torch.Tensor:
    data = torch.as_tensor(np.load(train_data_path), dtype=torch.float32)

    if data.ndim == 1:
        data = data[:, None]
    elif data.ndim > 2:
        data = data.reshape(data.shape[0], -1)

    if data.ndim != 2:
        raise ValueError(f"Expected 2D training data after flattening, got shape {tuple(data.shape)}")
    if not torch.isfinite(data).all():
        raise ValueError("Training data contains NaN or infinite values.")

    return data.contiguous()



class LightningPretrain(LightningModule):
    def __init__(self, model: TensorMLPModel):
        super().__init__()
        self.model = model

    def training_step(self, batch: tuple[torch.Tensor], batch_idx: int) -> torch.Tensor:
        (x,) = batch
        loss = self.model.train_loss(DDTensor(x)).mean()
        self.log("loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=0.0005)


def _train_lightning_model(model: TensorMLPModel, train_data: torch.Tensor, batch_size: int, max_epochs: int) -> None:
    train_data = train_data.to(model.device)
    dl = DataLoader(TensorDataset(train_data), batch_size=batch_size, shuffle=True)
    trainer = Trainer(max_epochs=max_epochs, logger=False, enable_checkpointing=False)
    trainer.fit(LightningPretrain(model), dl)


def train_flow(
    train_data: torch.Tensor,
    output_model_path: Path,
    device: torch.device,
    batch_size: int = 128,
    max_epochs: int = 10,
) -> Path:
    input_dim = train_data.shape[1]
    model = TensorMLPModel(
        OptimalTransportScheduler(),
        output_type="velocity",
        input_dim=input_dim,
        device=device,
    ).to(device)
    _train_lightning_model(model, train_data, batch_size, max_epochs)
    torch.save(model.model.state_dict(), output_model_path)



def train_diffusion(
    train_data: torch.Tensor,
    output_model_path: Path,
    device: torch.device,
    batch_size: int = 128,
    max_epochs: int = 10,
    beta_0: float = 0.1,
    beta_1: float = 12.0,
) -> Path:
    input_dim = train_data.shape[1]
    n_steps = 1000
    betas = torch.linspace(beta_0 / n_steps, beta_1 / n_steps, n_steps, device=device)
    scheduler = DiffusionScheduler((1.0 - betas).cumprod(dim=0))
    model = TensorMLPModel(
        scheduler,
        output_type="epsilon",
        input_dim=input_dim,
        device=device,
    ).to(device)
    _train_lightning_model(model, train_data, batch_size, max_epochs)
    torch.save(model.model.state_dict(), output_model_path)

if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data = load_training_data(args.train_data_path)
    
    output_model_path = Path(args.output_model_path)
    output_model_path.parent.mkdir(parents=True, exist_ok=True)


    print(f"Training {args.model_type} model -> {args.output_model_path}")
    if args.model_type == "flow":
        train_flow(
            train_data,
            output_model_path,
            device=device,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
        )
    else:
        train_diffusion(
            train_data,
            output_model_path,
            device=device,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            beta_0=args.beta_0,
            beta_1=args.beta_1,
        )
