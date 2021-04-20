from base.base_trainer import BaseTrainer
from base.base_dataset import BaseADDataset
from base.base_net import BaseNet
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import roc_auc_score

import logging
import time
import torch
import torch.optim as optim
import numpy as np


class DeepSVDDTrainer(BaseTrainer):

    def __init__(self, objective, R, c, nu: float, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 150,
                 lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
                 n_jobs_dataloader: int = 0, mode: str = 'original', add_params=None):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device,
                         n_jobs_dataloader)

        assert objective in ('one-class', 'soft-boundary'), "Objective must be either 'one-class' or 'soft-boundary'."
        self.objective = objective

        # Deep SVDD parameters
        self.R = torch.tensor(R, device=self.device)  # radius R initialized with 0 by default.
        self.c = torch.tensor(c, device=self.device) if c is not None else None
        self.c_g3 = None
        self.c_gi = None
        self.nu = nu

        # Optimization parameters
        self.warm_up_n_epochs = 10  # number of training epochs for soft-boundary Deep SVDD before radius R gets updated

        # Results
        self.train_time = None
        self.test_auc = None
        self.test_time = None
        self.test_scores = None
        self.mode = mode
        self.add_params = add_params

    def train(self, dataset: BaseADDataset, net: BaseNet):
        logger = logging.getLogger()

        # Set device for network
        net = net.to(self.device)

        # Get train data loader
        train_loader, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                               amsgrad=self.optimizer_name == 'amsgrad')

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        # Initialize hypersphere center c (if c not loaded)
        if self.c is None:
            logger.info('Initializing center c...')
            if 'cw' in self.add_params:
                self.c = self.init_center_c_w(train_loader, net)
            else:
                self.c = self.init_center_c(train_loader, net)
            # self.c_g2 = self.init_center_c_grad(train_loader, net, net.conv2.weight)
            old_mode = self.mode
            if self.mode == 'weight' or self.mode == 'both':
                self.mode = 'weight'
                self.c_g3 = self.init_center_c_grad(train_loader, net, None).detach()
                self.mode = old_mode
            if self.mode == 'input' or self.mode == 'both':
                self.mode = 'input'
                self.c_gi = self.init_center_c_grad(train_loader, net, None).detach()
                self.mode = old_mode
            # self.c_g2 = self.c_g2.detach()
            # self.c_g3 = self.c_g3.detach()
            logger.info('Center c initialized.')

        # Training
        center_update_epochs = 25
        if 'fast_c' in self.add_params:
            center_update_epochs = 5
        logger.info('Starting training...')
        start_time = time.time()
        for epoch in range(self.n_epochs):
            net.train()
            if epoch in self.lr_milestones:
                logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_last_lr()[0]))

            loss_epoch = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            for data in train_loader:
                inputs, _, _ = data
                inputs = inputs.to(self.device)

                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                inputs.requires_grad_(True)
                outputs, out_p = net(inputs, return_prev=True)
                loss3 = None
                old_mode = self.mode
                # grads2 = torch.autograd.grad(outputs=outputs.sum(), inputs=net.conv2.weight, create_graph=True, retain_graph=True)[0]
                if self.mode == 'weight' or self.mode == 'both':
                    self.mode = 'weight'
                    grads3 = torch.autograd.grad(outputs=outputs.sum(), inputs=None, create_graph=True,
                                                 retain_graph=True)[0]
                    dist3 = (grads3 - self.c_g3.expand_as(grads3)) ** 2
                    loss3 = torch.sum(dist3) / outputs.shape[0]
                    self.mode = old_mode
                if self.mode == 'input' or self.mode == 'both':
                    self.mode = 'input'
                    grads3 = \
                    torch.autograd.grad(outputs=outputs.sum(), inputs=inputs, create_graph=True, retain_graph=True)[0]
                    if 'grad_norm' in self.add_params:
                        grads3 = grads3 / (torch.sqrt(
                            torch.sum(grads3 ** 2, dim=tuple(range(1, len(grads3.shape))), keepdim=True)) + 1e-5)
                    dist3 = (grads3 - self.c_gi.expand_as(grads3)) ** 2
                    dist3 = torch.sum(dist3.view(dist3.shape[0], -1), dim=1)
                    if loss3 is None:
                        loss3 = torch.mean(dist3)
                    else:
                        loss3 = loss3 + torch.mean(dist3)
                    self.mode = old_mode
                inputs.requires_grad_(False)
                # if r is None:
                #    r = torch.randn((1,) + grads.shape[1:], device=self.device)
                # print(outputs.shape, self.c.shape, grads.shape, self.c_g.expand_as(grads).shape)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)

                # dist2 = (grads2 - self.c_g2.expand_as(grads2))**2
                # dist = torch.sum((outputs - self.c) ** 2, dim=1)
                if self.objective == 'soft-boundary':
                    scores = dist - self.R ** 2
                    loss = self.R ** 2 + (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
                else:
                    loss = torch.mean(dist)

                    # loss2 = torch.mean(dist2)
                    if loss3 is not None:
                        loss = loss + loss3
                loss.backward()
                optimizer.step()

                # Update hypersphere radius R on mini-batch distances
                if (self.objective == 'soft-boundary') and (epoch >= self.warm_up_n_epochs):
                    self.R.data = torch.tensor(get_radius(dist, self.nu), device=self.device)

                loss_epoch += loss.item()
                n_batches += 1
            scheduler.step()

            #             if 'update_c' in self.add_params and epoch % center_update_epochs == 0:
            #                 logger.info('Updating center c...')
            #                 self.c = self.init_center_c(train_loader, net)
            #                 #self.c_g2 = self.init_center_c_grad(train_loader, net, net.conv2.weight)
            #                 old_mode = self.mode
            #                 if self.mode == 'weight' or self.mode == 'both':
            #                     self.mode = 'weight'
            #                     self.c_g3 = self.init_center_c_grad(train_loader, net, layer.weight).detach()
            #                     self.mode = old_mode
            #                 if self.mode == 'input' or self.mode == 'both':
            #                     self.mode = 'input'
            #                     self.c_gi = self.init_center_c_grad(train_loader, net, layer.weight).detach()
            #                     self.mode = old_mode
            #                 logger.info('Center c updated.')

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            logger.info('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'
                        .format(epoch + 1, self.n_epochs, epoch_train_time, loss_epoch / n_batches))

        self.train_time = time.time() - start_time
        logger.info('Training time: %.3f' % self.train_time)

        logger.info('Finished training.')

        return net

    def test(self, dataset: BaseADDataset, net: BaseNet):
        logger = logging.getLogger()

        # Set device for network
        net = net.to(self.device)

        # Get test data loader
        _, test_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Testing
        logger.info('Starting testing...')
        start_time = time.time()
        idx_label_score = []
        net.eval()
        with torch.no_grad():
            for data in test_loader:
                inputs, labels, idx = data
                inputs = inputs.to(self.device)
                outputs = net(inputs)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                if self.objective == 'soft-boundary':
                    scores = dist - self.R ** 2
                else:
                    scores = dist

                # Save triples of (idx, label, score) in a list
                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))

        self.test_time = time.time() - start_time
        logger.info('Testing time: %.3f' % self.test_time)

        self.test_scores = idx_label_score

        # Compute AUC
        _, labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)

        self.test_auc = roc_auc_score(labels, scores)
        logger.info('Test set AUC: {:.2f}%'.format(100. * self.test_auc))
        with open("test2.txt", "a") as myfile:
            myfile.write('{:.2f}% \n'.format(100. * self.test_auc))

        logger.info('Finished testing.')

    def init_center_c(self, train_loader: DataLoader, net: BaseNet, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        n_samples = 0
        c = torch.zeros(net.rep_dim, device=self.device)

        net.eval()
        with torch.no_grad():
            for data in train_loader:
                # get the inputs of the batch
                inputs, _, _ = data
                inputs = inputs.to(self.device)
                outputs = net(inputs)
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)

        c /= n_samples

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c

    def init_center_c_w(self, train_loader: DataLoader, net: BaseNet, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        n_samples = 0
        c = torch.zeros(net.rep_dim, device=self.device)

        net.eval()
        grad_max = torch.tensor([-np.inf], device=self.device)

        for data in train_loader:
            # get the inputs of the batch
            inputs, _, _ = data
            inputs = inputs.to(self.device)
            inputs.requires_grad_(True)
            outputs = net(inputs)
            n_samples += outputs.shape[0]
            if self.mode == 'weight':
                pass
            #                 grads = torch.autograd.grad(outputs=outputs.sum(), inputs=layer, create_graph=True, retain_graph=True)[0]
            #                 grads = grads / (torch.sum(grads**2) + 1e-5)
            elif self.mode == 'input':
                grads = \
                torch.autograd.grad(outputs=outputs.sum(), inputs=inputs, create_graph=False, retain_graph=False)[0]
                b = grads.shape[0]
                grads_norm = (torch.sum(grads.view(b, -1) ** 2) + 1e-5)
                grad_max = torch.maximum(grad_max, grads_norm.max())

            inputs.requires_grad_(False)

        # with torch.no_grad():
        for data in train_loader:
            # get the inputs of the batch
            inputs, _, _ = data
            inputs = inputs.to(self.device)
            inputs.requires_grad_(True)
            outputs = net(inputs)
            if self.mode == 'weight':
                pass
            #                 grads = torch.autograd.grad(outputs=outputs.sum(), inputs=layer, create_graph=True, retain_graph=True)[0]
            #                 grads = grads / (torch.sum(grads**2) + 1e-5)
            elif self.mode == 'input':
                grads = \
                torch.autograd.grad(outputs=outputs.sum(), inputs=inputs, create_graph=False, retain_graph=False)[0]
                b = grads.shape[0]
                grads_norm = (torch.sum(grads.view(b, -1) ** 2) + 1e-5)
                outputs = (1 - grads_norm / grad_max) * outputs
            inputs.requires_grad_(False)
            n_samples += outputs.shape[0]
            c += torch.sum(outputs.detach(), dim=0)

        c /= n_samples

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c

    def init_center_c_grad(self, train_loader: DataLoader, net: BaseNet, layer: torch.nn.Module, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        n_samples = 0
        c = None

        net.eval()
        # with torch.no_grad():
        for data in train_loader:
            # get the inputs of the batch
            inputs, _, _ = data
            inputs = inputs.to(self.device)
            inputs.requires_grad_(True)
            outputs = net(inputs)
            n_samples += outputs.shape[0]
            if self.mode == 'weight':
                grads = torch.autograd.grad(outputs=outputs.sum(), inputs=layer, create_graph=True, retain_graph=True)[
                    0]
                grads = grads / (torch.sum(grads ** 2) + 1e-5)
            elif self.mode == 'input':
                grads = torch.autograd.grad(outputs=outputs.sum(), inputs=inputs, create_graph=True, retain_graph=True)[
                    0]
                if 'grad_norm' in self.add_params:
                    grads = grads / (torch.sqrt(
                        torch.sum(grads ** 2, dim=tuple(range(1, len(grads.shape))), keepdim=True)) + 1e-5)
                grads = torch.sum(grads, dim=0)
            inputs.requires_grad_(False)
            if c is None:
                c = torch.zeros_like(grads)
            c += grads.detach()
        c /= n_samples

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c


def get_radius(dist: torch.Tensor, nu: float):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)
