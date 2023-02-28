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

PAD = 0
START = 1
END = 2
UNK = 3


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


def validate(device, encoder, decoder, val_loader, criterion, loss_obj):

    references = []
    test_references = []
    hypotheses = []
    all_imgs = []
    all_alphas = []

    print("Started validation...")
    decoder.eval()
    encoder.eval()

    losses = loss_obj()

    num_batches = len(val_loader)
    # Batches
    for i, (imgs, caps, caplens) in enumerate(tqdm(val_loader)):

        imgs_jpg = imgs.numpy()
        imgs_jpg = np.swapaxes(np.swapaxes(imgs_jpg, 1, 3), 1, 2)

        # Forward prop.
        imgs = encoder(imgs.to(device))
        caps = caps.to(device)

        scores, caps_sorted, decode_lengths, alphas = decoder(
            imgs, caps, caplens)
        targets = caps_sorted[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        scores_packed = pack_padded_sequence(
            scores, decode_lengths, batch_first=True)[0]
        targets_packed = pack_padded_sequence(
            targets, decode_lengths, batch_first=True)[0]

        # Calculate loss
        loss = criterion(scores_packed, targets_packed)
        loss += ((1. - alphas.sum(dim=1)) ** 2).mean()
        losses.update(loss.item(), sum(decode_lengths))

        # References
        for j in range(targets.shape[0]):
            # validation dataset only has 1 unique caption per img
            img_caps = targets[j].tolist()
            clean_cap = [w for w in img_caps if w not in [
                PAD, START, END]]  # remove pad, start, and end
            img_captions = list(map(lambda c: clean_cap, img_caps))
            test_references.append(clean_cap)
            references.append(img_captions)

        # Hypotheses
        _, preds = torch.max(scores, dim=2)
        preds = preds.tolist()
        temp_preds = list()
        for j, p in enumerate(preds):
            pred = p[:decode_lengths[j]]
            pred = [w for w in pred if w not in [PAD, START, END]]
            temp_preds.append(pred)  # remove pads, start, and end
        preds = temp_preds
        hypotheses.extend(preds)

        if i == 0:
            all_alphas.append(alphas)
            all_imgs.append(imgs_jpg)
