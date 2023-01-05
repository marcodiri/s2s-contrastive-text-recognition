import argparse
import os
import string
from os import path

import torch
import torch.utils.data
import torchmetrics
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint, Timer)
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

import config
from dataset import WordsDataModule
from model import Decoder, Encoder, SeqCLRModule, TxtRecModule


def predict(opt):
    opt.valid_data = opt.eval_data
    data = WordsDataModule(opt)
    data.setup(stage="validate")
    # data.show_batch()
    dl = data.val_dataloader()
    
    encoder = Encoder(opt)
    decoder = Decoder(encoder.SequenceModeling_output, opt)
    asd1 = encoder.state_dict()
    asd2 = decoder.state_dict()
    model = TxtRecModule.load_from_checkpoint(
        opt.saved_model,
        encoder=encoder,
        decoder=decoder)
    model.hparams.opt.save_dir = opt.save_dir
    
    cer = torchmetrics.CharErrorRate()
    wer = torchmetrics.WordErrorRate()
    cumul_correct_words = 0
    num_preds_till_now = 0
    cumul_loss = 0
    for batch_idx, batch in enumerate(dl, start=1):
        # disable randomness, dropout, etc...
        model.eval()
        # predict with the model
        result = model.validation_step(batch, 0)
        preds_str = data.label_converter.decode(result["preds_index"], result["length_for_pred"])
        targets = data.label_converter.decode(result["target"], result["length_for_loss"])
        
        loss = result["loss"].detach().float()
        cumul_loss += loss
        print(f"loss: {loss}")
        print(f"cumul loss: {cumul_loss/batch_idx}")
        
        preds_ = result["preds"].contiguous().view(-1, result["preds"].shape[-1])
        target_ = result["target"].contiguous().view(-1)
        print(f"batch acc: {model.val_accuracy(preds_, target_)}")
        print(f"total acc: {model.val_accuracy.compute()}")
        
        pred_ws = []
        target_ws = []
        batch_size = batch[0].shape[0]
        batch_correct_words = 0
        for i,w in enumerate(targets):
            if '[s]' in preds_str[i]:
                pred_w = preds_str[i][:preds_str[i].index('[s]')]
            else:
                pred_w = preds_str[i]
            target_w = targets[i][:targets[i].index('[s]')]
            if pred_w == target_w:
                batch_correct_words += 1
            pred_ws.append(pred_w)
            target_ws.append(target_w)

        print(f"batch words acc: {batch_correct_words/batch_size}")
        cumul_correct_words += batch_correct_words
        num_preds_till_now += batch_size
        print(f"cumul words acc: {cumul_correct_words/num_preds_till_now}")
        
        print(f"batch cer: {cer(pred_ws, target_ws)}")
        cer.update(pred_ws, target_ws)
        print(f"total cer: {cer.compute()}")
        print(f"batch wer: {wer(pred_ws, target_ws)}")
        wer.update(pred_ws, target_ws)
        print(f"total wer: {wer.compute()}")
        
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', help='Where to store logs and models')
    parser.add_argument('--eval_data', required=True, help='path to evaluation dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--saved_model', required=True, help="path to saved_model to evaluation")
    parser.add_argument('--disable_cuda', action='store_true',
                        help='disable CUDA')
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    parser.add_argument('--data_filtering_off', action='store_true', help='for data_filtering_off mode')
    parser.add_argument('--baiduCTC', action='store_true', help='for data_filtering_off mode')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str, required=True, help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, required=True, help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, required=True, help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, required=True, help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
    
    opt = parser.parse_args()
    
    if not opt.exp_name:
        opt.exp_name = f'{opt.Transformation}-{opt.FeatureExtraction}-{opt.SequenceModeling}-{opt.Prediction}'
    
    opt.save_dir = "result"
    os.makedirs(path.join(opt.save_dir, opt.exp_name), exist_ok=True)
    
    if not opt.disable_cuda and torch.cuda.is_available():
        config.device = torch.device('cuda')
    else:
        config.device = torch.device('cpu')
    
    """ vocab / character number configuration """
    if opt.sensitive:
        opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).
    
    predict(opt)
