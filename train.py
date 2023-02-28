import pickle
import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence
from data_loader import get_loader
from nltk.translate.bleu_score import corpus_bleu
from processData import Vocabulary
from tqdm import tqdm
import torchvision.models as models
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import json
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import skimage.transform
from imageio import imread
from PIL import Image
import matplotlib.image as mpimg
from pytorch_pretrained_bert import BertTokenizer, BertModel
import imageio
from models.decoder import Decoder
from models.encoder import Encoder
from models.loss import loss_obj

##### HYPERPARAMS #####
grad_clip = 5.
num_epochs = 4
batch_size = 16
decoder_lr = 0.0004

glove_model = False
bert_model = True

from_checkpoint = True
train_model = True
valid_model = True

encoder_path = 'checkpoints/encoder_epoch4'
decoder_path = 'checkpoints/decoder_epoch4'


def init_model(device, mode='train'):
    # Clear CUDA cache
    torch.cuda.empty_cache()

    # Load Vocabulary
    print("Loading vocabulary...")
    with open('data/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
        print("Vocabulary loaded.")

    if mode == 'train':
        train_loader = get_loader('train', vocab, batch_size)
    elif mode == 'valid':
        train_loader = get_loader('val', vocab, batch_size)

    # Initialize models
    encoder = Encoder().to(device)
    decoder = Decoder(vocab, bert_model, device).to(device)
    losses = loss_obj()
    num_batches = len(train_loader)
    decoder_optimizer = torch.optim.Adam(
        params=decoder.parameters(), lr=decoder_lr)

    if from_checkpoint:
        print("Loading checkpoint...")
        decoder_checkpoint = torch.load(decoder_path)
        encoder.load_state_dict(torch.load(encoder_path))
        decoder.load_state_dict(decoder_checkpoint)
        decoder_optimizer.load_state_dict(
            decoder_checkpoint['decoder_optimizer'])
        print("Checkpoint loaded.")


def train(device, encoder, decoder, train_loader, decoder_optimizer, criterion, loss_obj):
    print("Started training...")
    for epoch in tqdm(range(num_epochs)):
        decoder.train()
        encoder.train()

        losses = loss_obj()
        num_batches = len(train_loader)

        for i, (imgs, caps, caplens) in enumerate(tqdm(train_loader)):

            imgs = encoder(imgs.to(device))
            caps = caps.to(device)

            scores, caps_sorted, decode_lengths, alphas = decoder(
                imgs, caps, caplens)
            scores = pack_padded_sequence(
                scores, decode_lengths, batch_first=True)[0]

            targets = caps_sorted[:, 1:]
            targets = pack_padded_sequence(
                targets, decode_lengths, batch_first=True)[0]

            loss = criterion(scores, targets).to(device)

            loss += ((1. - alphas.sum(dim=1)) ** 2).mean()

            decoder_optimizer.zero_grad()
            loss.backward()

            # grad_clip decoder
            for group in decoder_optimizer.param_groups:
                for param in group['params']:
                    if param.grad is not None:
                        param.grad.data.clamp_(-grad_clip, grad_clip)

            decoder_optimizer.step()

            losses.update(loss.item(), sum(decode_lengths))

            # save model each 1000 batches
            if i % 1000 == 0 and i != 0:
                print('epoch '+str(epoch+1)+'/4 ,Batch '+str(i) +
                      '/'+str(num_batches)+' loss:'+str(losses.avg))

                # adjust learning rate (create condition for this)
                for param_group in decoder_optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.8

                print('saving model...')

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': decoder.state_dict(),
                    'optimizer_state_dict': decoder_optimizer.state_dict(),
                    'loss': loss,
                }, './checkpoints/decoder_mid')

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': encoder.state_dict(),
                    'loss': loss,
                }, './checkpoints/encode_mid')

                print('model saved')

        torch.save({
            'epoch': epoch,
            'model_state_dict': decoder.state_dict(),
            'optimizer_state_dict': decoder_optimizer.state_dict(),
            'loss': loss,
        }, './checkpoints/decoder_epoch'+str(epoch+1))

        torch.save({
            'epoch': epoch,
            'model_state_dict': encoder.state_dict(),
            'loss': loss,
        }, './checkpoints/encoder_epoch'+str(epoch+1))

        print('epoch checkpoint saved')

    print("Completed training...")


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if train_model:
        encoder, decoder, train_loader, decoder_optimizer, criterion, loss_obj = init_model(
            device, mode='train')
        train(device, encoder, decoder, train_loader,
              decoder_optimizer, criterion, loss_obj)
