# __author__ = "Abhijeet Shrivastava"

import wandb
import torch as th
from torch import nn
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

import config
from dataset import TripletDataset
from model import Facenet


def train(config, log_writer):
    print('train start')

    # initializing
    model = Facenet(config.head, config.flatten, config.emb_dropout, config.emb_dim, config.pre_emb_param)
    if config.resume:
        model.load_state_dict(th.load(config.model_path))
    bestmodel = model
    best_valid = th.tensor(float('inf'))
    patience = config.patience

    loss_fn = nn.TripletMarginLoss(config.triplet_margin, p=2, reduction='sum')
    optimizer = th.optim.Adam(params=model.parameters(), lr=config.lr, eps=config.eps)
    # todo transofrmation
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = TripletDataset(hard_sampling=config.hard_sampling, model=model,
                                   selector_batch_size=config.selector_batch_size, root=config.data_dir, split='train',
                                   target_type='identity', transform=transform)
    val_dataset = TripletDataset(root=config.data_dir, split='valid', target_type='identity')

    # starting training
    for epoch in range(config.epochs):
        tol_loss = th.zeros([1])
        model.train()

        if config.hard_sampling:
            train_dataset.model = bestmodel
            train_dataloder = iter(
                train_dataset.find_hard_samples(train_dataset.unique_identities, config.train_batch_size))
        else:
            train_dataloder = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, )
            # num_workers=config.num_dataloader_worker)

        for i, triplet_batch in enumerate(train_dataloder):
            embeds = model(th.cat(triplet_batch))
            anchors = embeds[:config.train_batch_size]
            positives = embeds[config.train_batch_size:2 * config.train_batch_size]
            negatives = embeds[2 * config.train_batch_size:3 * config.train_batch_size]
            loss = loss_fn(anchors, positives, negatives)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss = loss.detach().numpy().item()
            print(f'step: {i} loss: {loss}')
            log_writer.add_scalar('loss/train_step', loss, i)
            tol_loss += loss

            for name, parm in model.named_parameters():
                log_writer.add_histogram('param/' + name, parm.data.cpu().numpy(), epoch)
                log_writer.add_histogram('grad/' + name, parm.grad.data.cpu().numpy(), epoch)
        log_writer.add_scalar('loss/tol_train', tol_loss, epoch)

        # evaluation loop
        if epoch % config.eval_interval == 0:
            tol_loss.zero_()
            model.eval()
            with th.no_grad():
                for i, triplet_batch in enumerate(DataLoader(val_dataset, batch_size=config.val_batch_size,
                                                             shuffle=True, num_workers=config.num_dataloader_worker)):
                    embeds = model(th.cat(triplet_batch))
                    anchors = embeds[:config.val_batch_size]
                    positives = embeds[config.val_batch_size:2 * config.val_batch_size]
                    negatives = embeds[2 * config.val_batch_size:3 * config.val_batch_size]
                    loss = loss_fn(anchors, positives, negatives)
                    loss = loss.detach().numpy().item()
                    tol_loss += loss
            log_writer.add_scalar('loss/tol_valid', tol_loss, epoch)
            # early stopping
            if tol_loss < best_valid:
                patience = config.patience
                th.save(model.state_dict(), 'models/model_%d.pkl' % config.exp_no)
            else:
                patience -= 1
            if patience == 0:
                print('early stoppping')
    print('training ended')
    return bestmodel


def test(model, log_writer, config):
    test_dataset = TripletDataset(root=config.data_dir, split='test', target_type='identity')
    tol_loss = th.zeros([1])
    loss_fn = nn.TripletMarginLoss(config.triplet_margin, p=2, reduction='sum')
    model.eval()
    with th.no_grad():
        for triplet_batch in DataLoader(test_dataset, batch_size=config.val_batch_size, shuffle=True):
            embeds = model(th.cat(triplet_batch))
            anchors = embeds[:config.val_batch_size]
            positives = embeds[config.val_batch_size:2 * config.val_batch_size]
            negatives = embeds[2 * config.val_batch_size:3 * config.val_batch_size]
            loss = loss_fn(anchors, positives, negatives)
            tol_loss += loss.detach().item()
            log_writer.add_embedding(th.cat([(anchors - positives) ** 2, (anchors - negatives) ** 2]),
                                     metadata=['p'] * len(positives) + ['n'] * len(negatives), tag='embedding dists')
    log_writer.add_scalar('loss/tol_test', tol_loss)


def test_lfw():
    # report performance metrics
    # draw recall and precision (ROC)curve with multiple threshold find threshold
    # log metrics
    pass


if __name__ == '__main__':
    # download dataset
    # TripletDataset(root=config.data_dir, split='all', target_type='identity', download=True)
    log_writer = SummaryWriter('logs/exp_%d' % config.exp_no)
    th.manual_seed(0)
    bestmodel = train(config=config,log_writer=log_writer)
    test(bestmodel, log_writer, config=config)
    test_lfw()  # todo report performance on lfw
    log_writer.close()
