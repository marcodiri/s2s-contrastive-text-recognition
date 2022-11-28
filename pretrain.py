import argparse
import os
import string
from os import path

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint, Timer)
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

import config
from dataset import WordsDataModule
from model import Encoder, SeqCLRModule
from model.augmentations import AugmentPreTraining


def train(opt):
    """ dataset preparation """
    if not opt.data_filtering_off:
        print('Filtering the images containing characters which are not in opt.character')
        print('Filtering the images whose label is longer than opt.batch_max_length')
        # see https://github.com/clovaai/deep-text-recognition-benchmark/blob/6593928855fb7abb999a99f428b3e4477d4ae356/dataset.py#L130

    opt.select_data = opt.select_data.split('-')
    opt.batch_ratio = opt.batch_ratio.split('-')
    
    data = WordsDataModule(opt, pretraining=True,
                           tranform=AugmentPreTraining(opt.imgH, opt.imgW))
    
    # Model creation/loading
    base_encoder = Encoder(opt)
    
    if opt.saved_model != '':
        if opt.FT:
            model = SeqCLRModule.load_from_checkpoint(
                opt.saved_model,
                strict=False,
                base_encoder=base_encoder)
        else:
            model = SeqCLRModule.load_from_checkpoint(
                opt.saved_model,
                base_encoder=base_encoder)
        model.hparams.opt.save_dir = opt.save_dir
    else:
        model = SeqCLRModule(base_encoder, opt)
    
    # Trainer setup + callbacks + loggers
    checkpoint_latest_path = path.join(
        opt.save_dir,
        opt.exp_name, 'latest')
    checkpoint_best_path = path.join(
        opt.save_dir,
        opt.exp_name, 'best')
    os.makedirs(checkpoint_latest_path, exist_ok=True)
    os.makedirs(checkpoint_best_path, exist_ok=True)
    
    checkpoint_best = ModelCheckpoint(
        dirpath=checkpoint_best_path,
        filename='{epoch}-{step}-{train_loss:.2f}',
        monitor='train_loss',
        mode='min',
        save_on_train_epoch_end=True
    )
    checkpoint_latest = ModelCheckpoint(
        dirpath=checkpoint_latest_path,
        filename='{epoch}-{step}',
        every_n_epochs=1,
        save_on_train_epoch_end=True
    )
        
    class TimerLog(Timer):
        def on_train_epoch_end(self, trainer: Trainer, *args, **kwargs):
            trainer.model.log("train_elapsed_time",
                              self.time_elapsed("train"))
    timer = TimerLog()
    
    lr_monitor = LearningRateMonitor()
    
    callbacks=[timer,
               lr_monitor,
               checkpoint_best,
               checkpoint_latest]
    
    if opt.earlystopping:
        patience = opt.patience*2-1
        early_stopping = EarlyStopping(
            monitor='train_loss',
            mode='min',
            patience=patience,
            verbose=True)
        callbacks.append(early_stopping)
        print(f"Enabled EarlyStopping with patience {patience}")
    
    logger_tb = TensorBoardLogger(
        save_dir=path.join(opt.save_dir, opt.exp_name, 'tb'))
    logger_csv = CSVLogger(
        save_dir=path.join(opt.save_dir, opt.exp_name, 'csv'))
    
    trainer = Trainer(
        accelerator="auto" if config.device.type=="cuda" else "cpu",
        devices=1,
        benchmark=True,
        max_steps=opt.num_iter,
        gradient_clip_val=opt.grad_clip,
        callbacks=callbacks,
        log_every_n_steps=50,
        logger=[logger_tb, logger_csv])
    
    if opt.saved_model != '':
        trainer.fit(model, data, ckpt_path=opt.saved_model)
    else:
        trainer.fit(model, data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', help='Where to store logs and models')
    parser.add_argument('--train_data', required=True, help='path to training dataset')
    parser.add_argument('--manualSeed', type=int, default=1111, help='for random seed setting')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--num_iter', type=int, default=300000, help='number of iterations to train for')
    parser.add_argument('--save_dir', default='saved_models', help="path where to save logs and checkpoints")
    parser.add_argument('--saved_model', default='', help="path to Lightning module to continue training; \
        base_model.pth must be present in the same directory")
    parser.add_argument('--disable_cuda', action='store_true',
                        help='disable CUDA')
    parser.add_argument('--FT', action='store_true', help='whether to do fine-tuning')
    parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is Adadelta)')
    parser.add_argument('--lr', type=float, default=1, help='learning rate, default=1.0 for Adadelta')
    parser.add_argument('--plateau', type=float, default=0.1, help='if != 0, \
        the learning rate will be reduced by the --plateau factor if \
        the validation loss does not improve for --patience rounds. \
        default=0.1')
    parser.add_argument('--patience', type=int, default=5, help='patience for the plateau scheduler')
    parser.add_argument('--earlystopping', action='store_true', help='Whether to use early stopping. \
        The patience will be set to --patience*2-1')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
    parser.add_argument('--rho', type=float, default=0.95, help='decay rate rho for Adadelta. default=0.95')
    parser.add_argument('--eps', type=float, default=1e-8, help='eps for Adadelta. default=1e-8')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping value. default=5')
    """ Data processing """
    parser.add_argument('--select_data', type=str, default='/',
                        help='select training data (default is /, which means all subfolders are used as training data)')
    parser.add_argument('--batch_ratio', type=str, default='1.0',
                        help='assign ratio for each selected data in the batch')
    parser.add_argument('--total_data_usage_ratio', type=str, default='1.0',
                        help='total data usage ratio, this ratio is multiplied to total number of data.')
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str,
                        default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    parser.add_argument('--data_filtering_off', action='store_true', help='for data_filtering_off mode')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str, required=True, help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, required=True,
                        help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, required=True, help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--InstanceMapping', type=str, required=True, help='Instance Mapping stage. WindowToInstance')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1,
                        help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
    parser.add_argument('--mapping_instances', type=int, default=5, help='the size of the output of the instance mapping')
    parser.add_argument('--temperature', type=float, default=0.07, help='the temperature parameter cor NT-Xent loss')

    opt = parser.parse_args()

    if not opt.exp_name:
        opt.exp_name = f'{opt.Transformation}-{opt.FeatureExtraction}-{opt.SequenceModeling}-{opt.InstanceMapping}'
        opt.exp_name += f'-Seed{opt.manualSeed}'
        # print(opt.exp_name)

    os.makedirs(path.join(opt.save_dir, opt.exp_name), exist_ok=True)
    
    if not opt.disable_cuda and torch.cuda.is_available():
        config.device = torch.device('cuda')
    else:
        config.device = torch.device('cpu')

    """ vocab / character number configuration """
    if opt.sensitive:
        # opt.character += 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    """ Seed and GPU setting """
    # print("Random Seed: ", opt.manualSeed)
    seed_everything(opt.manualSeed, workers=True)

    train(opt)