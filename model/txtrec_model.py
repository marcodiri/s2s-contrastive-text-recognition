from os import path
import re

import torch
import torch.optim as optim
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class TxtRecModule(pl.LightningModule):
    def __init__(self, encoder, decoder, opt):
        super().__init__()
        self.save_hyperparameters(ignore=["encoder", "decoder"])
        self.encoder = encoder
        self.decoder = decoder
        
        """ setup loss """
        if 'CTC' in self.hparams.opt.Prediction:
            self.train_accuracy = torchmetrics.Accuracy()
            self.val_accuracy = torchmetrics.Accuracy()
            if self.hparams.opt.baiduCTC:
                # need to install warpctc. see our guideline.
                from warpctc_pytorch import CTCLoss
                self.criterion = CTCLoss()
            else:
                self.criterion = nn.CTCLoss(zero_infinity=True)
        else:
            # ignore [GO] token = ignore index 0
            self.train_accuracy = torchmetrics.Accuracy(
                task='multiclass',
                num_classes=opt.num_class,
                ignore_index=0)
            self.val_accuracy = torchmetrics.Accuracy(
                task='multiclass',
                num_classes=opt.num_class,
                ignore_index=0)
            self.criterion = nn.CrossEntropyLoss(ignore_index=0)
            
    def training_step(self, batch, batch_idx):
        image, tensor_label, length = batch
        batch_size = image.size(0)
        if 'CTC' in self.hparams.opt.Prediction:
            features = self.encoder(image)
            preds = self.decoder(features, tensor_label)
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            if self.hparams.opt.baiduCTC:
                preds = preds.permute(1, 0, 2)  # to use CTCLoss format
                loss = self.criterion(preds, tensor_label, preds_size, length) / batch_size
            else:
                preds = preds.log_softmax(2).permute(1, 0, 2)
                loss = self.criterion(preds, tensor_label, preds_size, length)
            # TODO: compute accuracy for CTC
        else:
            features = self.encoder(image)
            preds = self.decoder(features, tensor_label[:, :-1])  # align with Attention.forward
            target = tensor_label[:, 1:]  # without [GO] Symbol
            
            preds = preds.view(-1, preds.shape[-1])
            target = target.contiguous().view(-1)
            loss = self.criterion(preds, target)
            self.train_accuracy.update(preds, target)
            
        return loss

    def validation_step(self, batch, batch_idx):
        image, text_for_loss, length_for_loss = batch
        device = text_for_loss.device
        batch_size = image.size(0)
        # For max length prediction
        length_for_pred = torch.IntTensor(
            [self.hparams.opt.batch_max_length] * batch_size).to(device)
        text_for_pred = torch.LongTensor(
            batch_size,
            self.hparams.opt.batch_max_length + 1).fill_(0).to(device)

        if 'CTC' in self.hparams.opt.Prediction:
            features = self.encoder(image)
            preds = self.decoder(features, text_for_pred)

            # Calculate evaluation loss for CTC deocder.
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            # permute 'preds' to use CTCloss format
            if self.hparams.opt.baiduCTC:
                loss = self.criterion(preds.permute(1, 0, 2), text_for_loss, preds_size, length_for_loss) / batch_size
            else:
                loss = self.criterion(preds.log_softmax(2).permute(1, 0, 2), text_for_loss, preds_size, length_for_loss)

            # Select max probabilty (greedy decoding)
            if self.hparams.opt.baiduCTC:
                _, preds_index = preds.max(2)
                preds_index = preds_index.view(-1)
            else:
                _, preds_index = preds.max(2)
            # TODO: compute accuracy for CTC
            return {"loss": loss,
                    "preds_index": preds_index,
                    "preds_size": preds_size}
        
        else:
            features = self.encoder(image)
            preds = self.decoder(features, text_for_pred, is_train=False)

            preds = preds[:, :text_for_loss.shape[1] - 1, :]
            target = text_for_loss[:, 1:]  # without [GO] Symbol
            
            # select max probabilty (greedy decoding)
            _, preds_index = preds.max(2)
            
            preds_ = preds.contiguous().view(-1, preds.shape[-1])
            target_ = target.contiguous().view(-1)
            loss = self.criterion(preds_, target_)
            self.val_accuracy.update(preds_, target_)
            return {"loss": loss,
                    "preds": preds,
                    "preds_index": preds_index,
                    "target": target,
                    "length_for_pred": length_for_pred,
                    "length_for_loss": length_for_loss}
    
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
                monitor: val_acc,
                reduction factor: {self.hparams.opt.plateau},
                patience: {self.hparams.opt.patience}}}""")
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=self.hparams.opt.plateau,
                min_lr=1e-4,
                threshold=1e-3,
                threshold_mode='abs',
                cooldown=self.hparams.opt.patience,
                verbose=True,
                patience=self.hparams.opt.patience)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_acc",
                    "frequency": self.hparams.opt.val_interval
                },
            }
        else:
            return optimizer
    
    def configure_callbacks(self):
        compute_metrics = self._ComputeMetrics()
        return [compute_metrics]
    
    class _ComputeMetrics(Callback):
        cumul_correct_words = 0
        num_preds_till_now = 0
        
        def on_train_batch_end(self, trainer, pl_module, outputs, batch,
                               batch_idx):
            pl_module.log("train_loss", outputs["loss"], prog_bar=False)
            
            if 'CTC' in pl_module.hparams.opt.Prediction:
                # TODO: log accuracy for CTC
                pass
            else:
                pl_module.log("train_acc", pl_module.train_accuracy,
                              on_step=False, on_epoch=True, prog_bar=False)
        
        def on_validation_batch_end(self, trainer, pl_module, outputs, batch,
                                    batch_idx, dataloader_idx):
            pl_module.log("val_loss", outputs["loss"],
                              on_step=False, on_epoch=True, prog_bar=False)
            
            converter = trainer.datamodule.label_converter
            self.labels = converter.decode(
                outputs["target"],
                outputs["length_for_loss"])
            if 'CTC' in pl_module.hparams.opt.Prediction:
                # TODO: log accuracy for CTC or ReduceLROnPlateau will crash
                self.preds_str = converter.decode(
                    outputs["preds_index"].data,
                    outputs["preds_size"].data)
            else:
                pl_module.log("val_acc", pl_module.val_accuracy,
                              on_step=False, on_epoch=True, prog_bar=True)
                self.preds_str = converter.decode(
                    outputs["preds_index"],
                    outputs["length_for_pred"])
            
            preds_prob = F.softmax(outputs["preds"], dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)
            
            batch_size = batch[0].shape[0]
            batch_correct_words = 0
            self.confidence_score_list = []
            for gt, pred, pred_max_prob in zip(self.labels, self.preds_str, preds_max_prob):
                if 'Attn' in pl_module.hparams.opt.Prediction:
                    gt = gt[:gt.find('[s]')]
                    pred_EOS = pred.find('[s]')
                    pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                    pred_max_prob = pred_max_prob[:pred_EOS]

                # To evaluate 'case sensitive model' with alphanumeric and case insensitve setting.
                if pl_module.hparams.opt.sensitive and pl_module.hparams.opt.data_filtering_off:
                    pred = pred.lower()
                    gt = gt.lower()
                    alphanumeric_case_insensitve = '0123456789abcdefghijklmnopqrstuvwxyz'
                    out_of_alphanumeric_case_insensitve = f'[^{alphanumeric_case_insensitve}]'
                    pred = re.sub(out_of_alphanumeric_case_insensitve, '', pred)
                    gt = re.sub(out_of_alphanumeric_case_insensitve, '', gt)

                # calculate confidence score (= multiply of pred_max_prob)
                try:
                    confidence_score = pred_max_prob.cumprod(dim=0)[-1]
                except:
                    confidence_score = 0  # for empty pred case, when prune after "end of sentence" token ([s])
                self.confidence_score_list.append(confidence_score)
                
                # calculate words accuracy
                if pred == gt:
                    batch_correct_words += 1
            self.cumul_correct_words += batch_correct_words
            self.num_preds_till_now += batch_size
        
        def on_validation_epoch_end(self, trainer, pl_module):
            pl_module.log("cumul_words_acc", self.cumul_correct_words/self.num_preds_till_now)
            self.cumul_correct_words = 0
            self.num_preds_till_now = 0
            
            # show some predicted results
            dashed_line = '-' * 80
            head = f'{"Ground Truth":25s} | {"Prediction":25s} | Confidence Score & T/F'
            predicted_result_log = f'{dashed_line}\n{head}\n{dashed_line}\n'
            for gt, pred, confidence in zip(self.labels[:5], self.preds_str[:5], self.confidence_score_list[:5]):
                if 'Attn' in pl_module.hparams.opt.Prediction:
                    gt = gt[:gt.find('[s]')]
                    pred = pred[:pred.find('[s]')]

                predicted_result_log += f'{gt:25s} | {pred:25s} | {confidence:0.4f}\t{str(pred == gt)}\n'
            predicted_result_log += f'{dashed_line}'
            
            with open(path.join(pl_module.hparams.opt.save_dir,
                                pl_module.hparams.opt.exp_name,
                                'log_train.txt'), 'a') as log:
                log.write(predicted_result_log + '\n')
