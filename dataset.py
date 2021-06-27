# __author__ = "Abhijeet Shrivastava"

import torch as th
from torch.utils.data import dataloader, dataset, DataLoader, Subset
from torchvision.datasets import CelebA


class TripletDataset(CelebA):
    # dataset, model, selector_batch_size, batch_size):
    def __init__(self, hard_sampling=False, model=None, selector_batch_size=None, **kwargs):
        super(TripletDataset, self).__init__(**kwargs)
        self.kwargs = kwargs
        self.hard_sampling = hard_sampling
        if self.hard_sampling:
            self.model = model
            self.selector_batch_size = selector_batch_size

        self.unique_identities = self.identity.unique()

        # fixing eval and test samples
        self.triplet_pair = {}
        if self.split == 'valid' or self.split == 'test':
            for idt in self.unique_identities:
                all_pos, _ = th.where(self.identity == idt)
                all_neg, _ = th.where(self.identity != idt)

                perm = th.randperm(len(all_pos))
                anchor = all_pos[perm[0]]
                if len(all_pos) > 2:  # todo filter data
                    pos = all_pos[perm[1]]
                else:
                    pos = anchor.clone()
                neg = all_neg[th.randperm(len(all_neg))[0]]

                self.triplet_pair[idt] = anchor, pos, neg

    def __getitem__(self, item):
        idt = self.identity[item]
        if self.hard_sampling and self.split == 'train':
            # toimpliment online sampling
            anchor, hard_pos, hard_neg = None, None, None
            triplet = anchor, hard_pos, hard_neg
            return triplet
        elif not self.hard_sampling and self.split == 'train':
            all_pos, _ = th.where(self.identity == idt)
            all_neg, _ = th.where(self.identity != idt)

            perm = th.randperm(len(all_pos))
            anchor, _ = CelebA.__getitem__(self, all_pos[perm[0]])
            pos, _ = CelebA.__getitem__(self, all_pos[perm[1]])
            neg, _ = CelebA.__getitem__(self, all_neg[th.randperm(len(all_neg))[0]])

            triplet = anchor, pos, neg
            return triplet

        elif self.split == 'valid' or self.split == 'test':
            anchor, pos, neg = self.triplet_pair[idt]
            anchor, _ = CelebA.__getitem__(self, anchor)
            pos, _ = CelebA.__getitem__(self, pos)
            neg, _ = CelebA.__getitem__(self, neg)

            triplet = anchor, pos, neg
            return triplet

    def find_hard_samples(self, identities, sampling_size):
        print('generating hard triplets...')
        self.model.eval()
        anchors, pos, neg = None, None, None
        for i, idt in enumerate(identities):
            hard_pos, hard_pos_dist = None, th.tensor(float('inf'))
            hard_neg, hard_neg_dist = None, th.tensor([-1.0])
            all_pos, _ = th.where(self.identity == idt)
            all_neg, _ = th.where(self.identity != idt)
            all_neg = th.where(sum(all_neg == i for i in identities))

            anchor = all_pos[[th.randperm(len(all_pos))[0]]]
            anchor, _ = CelebA.__getitem__(self, anchor)
            with th.no_grad():
                anchor = self.model(anchor.expand((2,*anchor.shape)))
            anchor = anchor[1:]
            ds = CelebA(**self.kwargs)
            pos_dataloader = DataLoader(Subset(ds, all_pos), shuffle=False, batch_size=self.selector_batch_size)
            neg_dataloader = DataLoader(Subset(ds, all_neg), shuffle=False, batch_size=self.selector_batch_size)
            for batch in pos_dataloader:
                with th.no_grad():
                    embds = self.model(batch[0])
                dists = th.sum((anchor - embds) ** 2, dim=1)
                val, ind = th.topk(dists, 2, largest=False)
                if val[1] < hard_pos_dist:
                    hard_pos_dist = val[1]  # since 0 will be distance from istself
                    hard_neg = batch[ind[1]]

            for batch in neg_dataloader:
                with th.no_grad():
                    embds = self.model(batch[0])
                dists = th.sum((anchor - embds) ** 2, dim=1)
                val, ind = th.topk(dists, 1, largest=True)
                if val[0] > hard_neg_dist:
                    hard_neg_dist = val[0]
                    hard_neg = batch[ind[0]]

            if anchors is None:
                anchors = th.zeros((0, *anchor.shape))
                pos = th.zeros((0, *anchor.shape))
                neg = th.zeros((0, *anchor.shape))
            anchors = th.stack(anchor, anchor, dim=0)
            pos = th.stack(pos, hard_pos, dim=0)
            neg = th.stack(neg, hard_neg, dim=0)
            if i % sampling_size == 0:
                yield anchors, pos, neg

