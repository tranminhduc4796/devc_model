import torch
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models.MF import RNNMatrixFactorization
from Dataset import Dataset
from pad_seq import pad_seq

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(loader, optimizer, model, criterion, epoch, lr_scheduler=None, logger=None):
    train_loss = 0
    progress_bar = tqdm(loader)
    for i, batch in enumerate(progress_bar):
        hours, users, shops, previous_shops_batch, rating = batch

        # zero the parameter gradients which is accumulated every backward pass
        optimizer.zero_grad()

        score = model(hours, users, shops, previous_shops_batch)
        score = score.squeeze()
        target = rating.squeeze()
        loss = criterion(score, target)

        loss.backward()
        optimizer.step()
        if lr_scheduler:
            lr_scheduler.step(epoch)

        # Print the progress
        train_loss += loss.item()
        avg_train_loss = train_loss / (i + 1)
        progress_bar.set_description("Train Loss: {}".format(avg_train_loss))
        if logger:
            for param_group in optimizer.param_groups:
                logger.add_scalar("lr", param_group['lr'], epoch)
            logger.add_scalar('train', avg_train_loss, epoch)
        progress_bar.refresh()


def valid(loader, model, criterion, epoch, logger=None):
    valid_losses = []
    progress_bar = tqdm(loader)
    for _, test_batch in enumerate(progress_bar):
        pri_idx_xpaths, rev_idx_xpaths, pri_lens, rev_lens, pri_ele_emb, rev_ele_emb, target = test_batch

        output = model(pri_idx_xpaths, rev_idx_xpaths, pri_lens, rev_lens, pri_ele_emb, rev_ele_emb)
        score = torch.sigmoid(output)
        target = target.view(1, -1, 1)
        valid_loss = criterion(score, target)
        valid_losses.append(valid_loss.data.cpu())

        mean_valid_loss = np.mean(valid_losses)
        progress_bar.set_description('Valid Loss: {}'.format(mean_valid_loss))
        if logger:
            logger.add_scalar('valid', mean_valid_loss, epoch)


def main():
    logger = SummaryWriter()

    data_dir = 'data_script'
    filename = 'data_for_model.csv'

    num_workers = 4  # Workers for CPU training.
    n_epoch = 2
    batch_size = 128
    weight_decay = 0.005
    momentum = 0.9

    n_user = 4447
    n_shop = 369

    # Loader for training data
    train_set = Dataset(data_dir, filename, 'train')
    train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size,
                              num_workers=0 if torch.cuda.is_available() else num_workers, collate_fn=pad_seq)

    # Loader for testing data
    test_set = Dataset(data_dir, filename, 'test')
    test_loader = DataLoader(test_set, batch_size=batch_size,
                             num_workers=0 if torch.cuda.is_available() else num_workers, collate_fn=pad_seq)

    # Create model
    model = RNNMatrixFactorization(n_user=n_user, n_shop=n_shop).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.5, weight_decay=weight_decay, momentum=momentum)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader), eta_min=0.000001)
    criterion = torch.nn.MSELoss()
    valid_criterion = torch.nn.MSELoss()

    for epoch in range(n_epoch):
        print('Epoch', epoch)
        # Training step
        train(loader=train_loader, model=model, optimizer=optimizer, criterion=criterion, epoch=epoch,
              lr_scheduler=lr_scheduler, logger=logger)

        # Valid step
        valid(loader=test_loader, model=model, criterion=valid_criterion, epoch=epoch, logger=logger)
    logger.close()
    torch.save(model.state_dict(), './models/BaseModel.pth')
    print('Finish!!!')


if __name__ == '__main__':
    main()
