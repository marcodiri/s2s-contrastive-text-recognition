import argparse
import os
import string
from os import path

import torch
import torch.utils.data
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Timer
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from dataset import WordsDataModule
from model import BaseModel, TxtRecModule


def train(opt):
    """ dataset preparation """
    if not opt.data_filtering_off:
        print('Filtering the images containing characters which are not in opt.character')
        print('Filtering the images whose label is longer than opt.batch_max_length')
        # see https://github.com/clovaai/deep-text-recognition-benchmark/blob/6593928855fb7abb999a99f428b3e4477d4ae356/dataset.py#L130

    opt.select_data = opt.select_data.split('-')
    opt.batch_ratio = opt.batch_ratio.split('-')
    
    data = WordsDataModule(opt)
    
    # Model creation/loading
    base = BaseModel(opt)
    if opt.saved_model != '':
        base_path = '/'.join(opt.saved_model.split('/')[:-1])
        base_path += "/base_model.pth"
        print(f'Loading pretrained model from {base_path}')
        if opt.FT:
            base.load_state_dict(torch.load(base_path), strict=False)
        else:
            base.load_state_dict(torch.load(base_path))
        model = TxtRecModule.load_from_checkpoint(opt.saved_model, model=base)
        model.hparams.opt.save_dir = opt.save_dir
    else:
        model = TxtRecModule(base, opt)
    
    # Trainer setup + callbacks + loggers
    checkpoint_latest_path = path.join(
        opt.save_dir,
        model.hparams.opt.exp_name, 'latest')
    checkpoint_best_path = path.join(
        opt.save_dir,
        model.hparams.opt.exp_name, 'best')
    os.makedirs(checkpoint_latest_path, exist_ok=True)
    os.makedirs(checkpoint_best_path, exist_ok=True)
    
    class BaseModelCheckpoint(ModelCheckpoint):
        def _save_checkpoint(self, trainer, filepath):
            # override base function to also save BaseModel
            torch.save(trainer.model.model.state_dict(), 
                       path.join(self.dirpath,
                                 'base_model.pth'))
            super()._save_checkpoint(trainer, filepath)
    checkpoint_best = BaseModelCheckpoint(
        dirpath=checkpoint_best_path,
        filename='{epoch}-{step}-{train_loss:.2f}-{val_loss:.2f}',
        monitor='val_loss',
        mode='min',
        save_on_train_epoch_end=False
    )
    checkpoint_latest = BaseModelCheckpoint(
        dirpath=checkpoint_latest_path,
        filename='{epoch}-{step}',
        every_n_epochs=1,
        save_on_train_epoch_end=True
    )
        
    class TimerLog(Timer):
        def on_train_epoch_end(self, trainer: Trainer, *args, **kwargs):
            self.log("train_elapsed_time", self.time_elapsed("train"))
    timer = TimerLog()
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=opt.patience)
    
    logger_tb = TensorBoardLogger(
        save_dir=path.join(opt.save_dir, model.hparams.opt.exp_name, 'tb'))
    logger_csv = CSVLogger(
        save_dir=path.join(opt.save_dir, model.hparams.opt.exp_name, 'csv'))
    
    trainer = Trainer(
        accelerator="auto",
        devices="auto",
        benchmark=True,
        max_steps=opt.num_iter,
        check_val_every_n_epoch=None,
        val_check_interval=opt.val_interval,
        gradient_clip_val=opt.grad_clip,
        callbacks=[timer,
                   early_stopping,
                   checkpoint_best,
                   checkpoint_latest],
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
    parser.add_argument('--valid_data', required=True, help='path to validation dataset')
    parser.add_argument('--manualSeed', type=int, default=1111, help='for random seed setting')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--num_iter', type=int, default=300000, help='number of iterations to train for')
    parser.add_argument('--val_interval', type=int, default=2000, help='Interval between each validation')
    parser.add_argument('--save_dir', default='saved_models', help="path where to save logs and checkpoints")
    parser.add_argument('--saved_model', default='', help="path to Lightning module to continue training; \
        base_model.pth must be present in the same directory")
    parser.add_argument('--FT', action='store_true', help='whether to do fine-tuning')
    parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is Adadelta)')
    parser.add_argument('--lr', type=float, default=1, help='learning rate, default=1.0 for Adadelta')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
    parser.add_argument('--rho', type=float, default=0.95, help='decay rate rho for Adadelta. default=0.95')
    parser.add_argument('--eps', type=float, default=1e-8, help='eps for Adadelta. default=1e-8')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping value. default=5')
    parser.add_argument('--baiduCTC', action='store_true', help='for data_filtering_off mode')
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
    parser.add_argument('--Prediction', type=str, required=True, help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1,
                        help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
    parser.add_argument('--patience', type=int, default=5, help='patience for the early stopping')

    opt = parser.parse_args()

    if not opt.exp_name:
        opt.exp_name = f'{opt.Transformation}-{opt.FeatureExtraction}-{opt.SequenceModeling}-{opt.Prediction}'
        opt.exp_name += f'-Seed{opt.manualSeed}'
        # print(opt.exp_name)

    os.makedirs(path.join(opt.save_dir, opt.exp_name), exist_ok=True)

    """ vocab / character number configuration """
    if opt.sensitive:
        # opt.character += 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    """ Seed and GPU setting """
    # print("Random Seed: ", opt.manualSeed)
    seed_everything(opt.manualSeed, workers=True)

    opt.num_gpu = torch.cuda.device_count()
    # print('device count', opt.num_gpu)
    if opt.num_gpu > 1:
        print('------ Use multi-GPU setting ------')
        print('if you stuck too long time with multi-GPU setting, try to set --workers 0')
        # check multi-GPU issue https://github.com/clovaai/deep-text-recognition-benchmark/issues/1
        opt.workers = opt.workers * opt.num_gpu
        opt.batch_size = opt.batch_size * opt.num_gpu

    train(opt)