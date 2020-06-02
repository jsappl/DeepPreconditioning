"""Training loop for PrecondNet based on OpenFOAM system matrix data set."""
from datetime import datetime

import torch
from spconv import SparseConvTensor
from torch.utils.tensorboard import SummaryWriter

from config import CFG
from data_loader import init_loaders
from loss import condition_loss
from model import PrecondNet
from test import test


def _train_epoch(loader, model, criterion, optimizer, device):
    """Train model for one epoch only."""
    model.train()
    epoch_loss = 0.
    for features, coors, shape, l_matrix in loader:
        sp_tensor = SparseConvTensor(
            features.T.to(device), coors.int().squeeze(), shape, 1)
        l_matrix = l_matrix[0].to(device)

        optimizer.zero_grad()
        loss = criterion(l_matrix, model(sp_tensor))
        loss.backward()
        optimizer.step()

        epoch_loss += loss.data.item()
    return epoch_loss, model


def _validate(loader, model, criterion, device):
    """Validate model during training."""
    model.eval()
    val_loss = 0.
    for features, coors, shape, l_matrix in loader:
        sp_tensor = SparseConvTensor(
            features.T.to(device), coors.int().squeeze(), shape, 1)
        l_matrix = l_matrix[0].to(device)

        loss = criterion(l_matrix, model(sp_tensor))
        val_loss += loss.data.item()
    return val_loss


def _train(model, criterion, optimizer, writer, device):
    """Training loop for PrecondNet."""
    train_loader, val_loader, _ = init_loaders()
    save_name = datetime.now().strftime('./runs/%b%d_%H-%M-%S'+'.pt')
    for epoch in range(CFG['N_EPOCHS']):
        epoch_loss, model = _train_epoch(
            train_loader, model, criterion, optimizer, device)
        torch.save(model.state_dict(), save_name)
        writer.add_scalar('Loss/train', epoch_loss/len(train_loader), epoch)

        if CFG['VALIDATE'] and epoch % 5 == 0:
            val_loss = _validate(val_loader, model, criterion, device)
            writer.add_scalar('Loss/val', val_loss/len(val_loader), epoch)
    return True


def main():
    torch.manual_seed(CFG['SEED'])
    torch.set_num_threads(CFG['N_THREADS'])
    device = torch.device(CFG['DEVICE'])

    model = PrecondNet().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    writer = SummaryWriter()
    writer.add_hparams(CFG, metric_dict={})

    _train(model, condition_loss, optimizer, writer, device)
    test(model, writer, device)

    writer.close()
    return True


if __name__ == '__main__':
    main()
