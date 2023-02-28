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

encoder_path = 'checkpoints_bert/encoder_epoch4'
decoder_path = 'checkpoints_bert/decoder_epoch4'

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

    if from_checkpoint:

        encoder = Encoder().to(device)
        decoder = Decoder(vocab, use_bert=bert_model, device='cuda').to(device)

        if torch.cuda.is_available():
            if bert_model:
                print('Pre-Trained BERT Model')
                encoder_checkpoint = torch.load('./checkpoints/encoder_epoch4')
                decoder_checkpoint = torch.load('./checkpoints/decoder_epoch4')
            elif glove_model:
                print('Pre-Trained GloVe Model')
                encoder_checkpoint = torch.load('./checkpoints/encoder_glove')
                decoder_checkpoint = torch.load('./checkpoints/decoder_glove')
            else:
                print('Pre-Trained Baseline Model')
                encoder_checkpoint = torch.load('./checkpoints/encoder_epoch2')
                decoder_checkpoint = torch.load('./checkpoints/encoder_epoch2')
        else:
            if bert_model:
                print('Pre-Trained BERT Model')
                encoder_checkpoint = torch.load(
                    './checkpoints/encoder_bert', map_location='cpu')
                decoder_checkpoint = torch.load(
                    './checkpoints/decoder_bert', map_location='cpu')
            elif glove_model:
                print('Pre-Trained GloVe Model')
                encoder_checkpoint = torch.load(
                    './checkpoints/encoder_glove', map_location='cpu')
                decoder_checkpoint = torch.load(
                    './checkpoints/decoder_glove', map_location='cpu')
            else:
                print('Pre-Trained Baseline Model')
                encoder_checkpoint = torch.load(
                    './checkpoints/encoder_baseline', map_location='cpu')
                decoder_checkpoint = torch.load(
                    './checkpoints/decoder_baseline', map_location='cpu')

        encoder.load_state_dict(encoder_checkpoint['model_state_dict'])
        decoder_optimizer = torch.optim.Adam(
            params=decoder.parameters(), lr=decoder_lr)
        decoder.load_state_dict(decoder_checkpoint['model_state_dict'])
        decoder_optimizer.load_state_dict(
            decoder_checkpoint['optimizer_state_dict'])

    else:
        encoder = Encoder().to(device)
        decoder = Decoder(vocab_size=len(vocab),
                          use_glove=glove_model, use_bert=bert_model).to(device)
        decoder_optimizer = torch.optim.Adam(
            params=decoder.parameters(), lr=decoder_lr)
        print('Model Initialized')

    return vocab, train_loader, encoder, decoder, decoder_optimizer


def print_sample(vocab, hypotheses, references, test_references, imgs, alphas, k, show_att=False):

    img_dim = 336  # 14*24

    hyp_sentence = []
    for word_idx in hypotheses[k]:
        hyp_sentence.append(vocab.idx2word[word_idx])

    ref_sentence = []
    for word_idx in test_references[k]:
        ref_sentence.append(vocab.idx2word[word_idx])

    print('Hypotheses: '+" ".join(hyp_sentence))
    print('References: '+" ".join(ref_sentence))

    img = imgs[0][k]
    imageio.imwrite('img.jpg', img)

    if show_att:
        image = Image.open('img.jpg')
        image = image.resize([img_dim, img_dim], Image.LANCZOS)
        for t in range(len(hyp_sentence)):

            plt.subplot(np.ceil(len(hyp_sentence) / 5.), 5, t + 1)

            plt.text(0, 1, '%s' % (
                hyp_sentence[t]), color='black', backgroundcolor='white', fontsize=12)
            plt.imshow(image)
            current_alpha = alphas[0][t, :].detach().numpy()
            alpha = skimage.transform.resize(current_alpha, [img_dim, img_dim])
            if t == 0:
                plt.imshow(alpha, alpha=0)
            else:
                plt.imshow(alpha, alpha=0.7)
            plt.axis('off')
    else:
        img = imageio.imread('img.jpg')
        plt.imshow(img)
        plt.axis('off')
        plt.show()


def generate_caption(vocab, device, encoder, decoder, val_loader):

    references = []
    test_references = []
    hypotheses = []
    all_imgs = []
    all_alphas = []

    print("Started generation...")
    decoder.eval()
    encoder.eval()

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

        print_sample(vocab, hypotheses, references, test_references,
                     all_imgs, all_alphas, i, False)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab, val_loader, encoder, decoder, decoder_optimizer = init_model(
        device, mode='train')
    generate_caption(vocab, device, encoder, decoder, val_loader)
