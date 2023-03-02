import os
import nltk
import torch
import torch.utils.data as data
from PIL import Image
from pycocotools.coco import COCO
from torchvision import transforms


class DataLoader(data.Dataset):
    def __init__(self, root, json, vocab, transform=None):

        self.root = root
        self.coco = COCO(json)
        self.ids = list(self.coco.anns.keys())
        self.vocab = vocab
        self.transform = transform

    def __getitem__(self, index):
        coco = self.coco
        vocab = self.vocab
        ann_id = self.ids[index]
        caption = coco.anns[ann_id]['caption']
        img_id = coco.anns[ann_id]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name']

        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target

    def __len__(self):
        return len(self.ids)

# Create another dataloader for the demo which loads a single or multiple images without captions


class DataLoaderDemo(data.Dataset):
    def __init__(self, root, vocab, transform=None):

        self.root = root
        self.vocab = vocab
        self.transform = transform
        self.ids = list(range(len(os.listdir(root))))

    def __getitem__(self, index):
        vocab = self.vocab
        path = os.listdir(self.root)[index]
        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, path

    def __len__(self):
        return len(self.ids)


def collate_fn(data):
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    images = torch.stack(images, 0)

    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return images, targets, lengths


def collate_fn_demo(data):
    images, paths = zip(*data)
    images = torch.stack(images, 0)
    return images, paths


def get_loader(method, vocab, batch_size):

    # train/validation paths
    if method == 'train':
        root = 'data/train2014_resized'
        json = 'data/annotations/captions_train2014.json'
    elif method == 'val':
        root = 'data/val2014_resized'
        json = 'data/annotations/captions_val2014.json'
    elif method == 'demo':
        root = 'data/demo'

    # rasnet transformation/normalization
    transform = transforms.Compose([
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    if method == 'demo':
        coco = DataLoaderDemo(root=root, vocab=vocab, transform=transform)
        data_loader = torch.utils.data.DataLoader(dataset=coco,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=1,
                                                  collate_fn=collate_fn_demo)
    else:
        coco = DataLoader(root=root, json=json,
                          vocab=vocab, transform=transform)
        data_loader = torch.utils.data.DataLoader(dataset=coco,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=1,
                                                  collate_fn=collate_fn)

    return data_loader
