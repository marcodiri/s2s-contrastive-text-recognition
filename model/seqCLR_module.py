import re
from os import path

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchmetrics
from pytorch_lightning.callbacks import Callback

from model.instance_mapping import WindowToInstance


class SeqCLRModule(pl.LightningModule):
    def __init__(self, base_encoder, opt):
        super().__init__()
        self.save_hyperparameters(ignore=["base_encoder"])
        self.base_encoder = base_encoder
        
        if opt.InstanceMapping == "WindowToInstance":
            self.instance_map = WindowToInstance(opt.mapping_instances)
        elif opt.InstanceMapping == "None":
            self.instance_map = None
        else:
            raise ValueError('Only WindowToInstance InstanceMapping is supported')

        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, X):
        Z = self.base_encoder(X)
        
        if self.instance_map:
            Z = self.instance_map(Z)
        
        # aggregate columns; corresponding frames (positive pairs)
        # will be at (batch_size*opt.mapping_instances) distance
        Z = Z.flatten(end_dim=1)
        
        return Z
    
    def compute_loss(self, frames):
        """Calculate NT-Xent loss.
        ref: https://github.com/google-research/simclr/blob/2fc637bdd6a723130db91b377ac15151e01e4fc2/tf2/objective.py#L35
        """
        
        # cos similarity matrix [batch_size*opt.mapping_instances*2];
        # Example:
        # sim_mat[0,0] = cos_sim(z0_a, z0_a)
        # sim_mat[0,1] = cos_sim(z0_a, z1_a)
        # sim_mat[0,batch_size*opt.mapping_instances] = cos_sim(z0_a, z0_b)
        # sim_mat[batch_size*opt.mapping_instances,0] = cos_sim(z0_b, z0_a)
        frames = nn.functional.normalize(frames, dim=-1)
        sim_mat = torch.mm(frames, frames.transpose(0,1))
        sim_mat /= self.hparams.opt.temperature
        
        # mask out self cosine similarity
        mask = torch.eye(
            sim_mat.shape[0],
            dtype=torch.bool,
            device=sim_mat.device)
        # big negative to remove self from cross entropy denominator
        sim_mat.masked_fill_(mask, -9e15)
        
        # find positive pair: (batch_size*opt.mapping_instances) away
        # each row is a zX_y and is True at the corrisponding peer column
        # Example:
        # sim_mat[0,0] = False
        # sim_mat[0,1] = False
        # sim_mat[0,batch_size*opt.mapping_instances] = True
        # sim_mat[batch_size*opt.mapping_instances,0] = True
        mask = mask.roll(
            shifts=sim_mat.shape[0]//2,
            dims=0)
        # convert to indices for CrossEntropy
        labels=mask.nonzero(as_tuple=True)[1]
        
        return self.criterion(sim_mat, labels)

    def training_step(self, batch, batch_idx):
        Z = self(batch)
        loss = self.compute_loss(Z)
        return loss
        
    def configure_optimizers(self):
        filtered_parameters = filter(lambda p: p.requires_grad,
                                     self.parameters())
        if self.hparams.opt.adam:
            optimizer = optim.Adam(
                filtered_parameters, 
                lr=self.hparams.opt.lr, 
                betas=(self.hparams.opt.beta1, 0.999))
        else:
            optimizer = optim.Adadelta(
                filtered_parameters, 
                lr=self.hparams.opt.lr, 
                rho=self.hparams.opt.rho, 
                eps=self.hparams.opt.eps)
        print("Optimizer:")
        print(optimizer)
        if self.hparams.opt.plateau:
            print(f"""Enabled ReduceLROnPlateau scheduler: {{
                monitor: train_loss,
                reduction factor: {self.hparams.opt.plateau},
                patience: {self.hparams.opt.patience}}}""")
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=self.hparams.opt.plateau,
                min_lr=1e-4,
                cooldown=self.hparams.opt.patience,
                verbose=True,
                patience=self.hparams.opt.patience)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "train_loss",
                },
            }
        else:
            return optimizer
    
    def configure_callbacks(self):
        compute_metrics = self._ComputeMetrics()
        return [compute_metrics]
    
    class _ComputeMetrics(Callback):
        def on_train_batch_end(self, trainer, pl_module, outputs, batch,
                               batch_idx):
            pl_module.log("train_loss", outputs["loss"], prog_bar=False)
        