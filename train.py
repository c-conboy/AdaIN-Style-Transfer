import argparse
from pathlib import Path
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
from PIL import Image, ImageFile
from tensorboardX import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
from matplotlib import pyplot as plt
import os


import AdaIN_net

cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
# Disable OSError: image file is truncated
ImageFile.LOAD_TRUNCATED_IMAGES = True


def InfiniteSampler(n):
    # i = 0
    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed()
            order = np.random.permutation(n)
            i = 0

class InfiniteSamplerWrapper(data.sampler.Sampler):
    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(InfiniteSampler(self.num_samples))

    def __len__(self):
        return 2 ** 31

def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.paths = list(Path(self.root).glob('*'))
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(str(path)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'


def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('-content_dir', type=str, required=False,
                    help='Directory path to a batch of content images')
parser.add_argument('-style_dir', type=str, required=False,
                    help='Directory path to a batch of style images')
parser.add_argument('-l', type=str, default='models/vgg_normalised.pth')

# training options
parser.add_argument('-s', default='./experiments',
                    help='Directory to save the model')
parser.add_argument('-log_dir', default='./logs',
                    help='Directory to save the log')
parser.add_argument('-lr', type=float, default=1e-4)
parser.add_argument('-lr_decay', type=float, default=5e-5)
parser.add_argument('-e', type=int, default=20)
parser.add_argument('-b', type=int, default=8)
parser.add_argument('-gamma', type=float, default=10.0)
parser.add_argument('-n_threads', type=int, default=1)
parser.add_argument('-cuda', type=str, default='N')
parser.add_argument('-p', type=str, default='decoder.png')
args = parser.parse_args()

if(args.cuda == 'Y'):
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

save_dir = Path(args.s)
log_dir = Path(args.log_dir)
log_dir.mkdir(exist_ok=True, parents=True)
writer = SummaryWriter(log_dir=str(log_dir))

decoder = AdaIN_net.encoder_decoder.decoder
vgg = AdaIN_net.encoder_decoder.encoder

vgg.load_state_dict(torch.load(args.l))
vgg = nn.Sequential(*list(vgg.children())[:31])
network = AdaIN_net.AdaIN_net(vgg, decoder)
network.train()
network.to(device)

content_tf = train_transform()
style_tf = train_transform()

content_dataset = FlatFolderDataset(args.content_dir, content_tf)
style_dataset = FlatFolderDataset(args.style_dir, style_tf)

content_iter = iter(data.DataLoader(
    content_dataset, batch_size=args.b,
    sampler=InfiniteSamplerWrapper(content_dataset)))
    #shuffle=True))
    #num_workers=args.n_threads))
style_iter = iter(data.DataLoader(
    style_dataset, batch_size=args.b,
    sampler=InfiniteSamplerWrapper(content_dataset)))
    #shuffle=True))
    #num_workers=args.n_threads))

optimizer = torch.optim.Adam(network.decoder.parameters(), lr=args.lr)

content_loss = torch.zeros(args.e)
style_loss = torch.zeros(args.e)

for i in tqdm(range(args.e)):
    adjust_learning_rate(optimizer, iteration_count=i)
    content_images = next(content_iter).to(device)
    style_images = next(style_iter).to(device)
    loss_c, loss_s = network.forward(content_images, style_images)
    loss = loss_c + (args.gamma * loss_s)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    writer.add_scalar('loss_content', loss_c.item(), i + 1)
    writer.add_scalar('loss_style', loss_s.item(), i + 1)

    if (i + 1) == args.e:
        state_dict = network.decoder.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict, 
                   './decoder_iter_{:d}.pth.tar'.format(i + 1))
    content_loss[i] = (loss_c.detach().item() / len(content_images))
    style_loss[i] = (loss_s.detach().item() / len(style_images))
writer.close()


state_dict = network.decoder.state_dict()
decoder_state_dict_file = os.path.join(os.getcwd(), args.s)

torch.save(state_dict, decoder_state_dict_file)

plt.plot(content_loss)
plt.plot(style_loss)
plt.plot([x + y for x, y in zip(content_loss, style_loss)])


plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig(args.p)