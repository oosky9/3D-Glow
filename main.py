from tqdm import tqdm
import numpy as np
from math import log

import statistics
import argparse
import os
import glob

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import utils

from model import Glow

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_MIN_INTENSITY = 0
DATA_MAX_INTENSITY = 667

def pre(x):
    return(x - DATA_MIN_INTENSITY) / (DATA_MAX_INTENSITY - DATA_MIN_INTENSITY)

def post(x):
    return x * (DATA_MAX_INTENSITY - DATA_MIN_INTENSITY) + DATA_MIN_INTENSITY

def get_model_name(p):
    file_names = glob.glob(os.path.join(p, "model*.pt"))
    return file_names[-1]

# "mode = ['train', 'valid', 'test']"
def sample_data(path, batch_size, mode, shuffle):
    np_data = np.load(os.path.join(path, mode + ".npy"))
    np_data = pre(np_data)

    if len(np_data.shape) == 4:
        dataset = torch.tensor(np_data, dtype=torch.float32).unsqueeze(1)
    else:
        dataset = torch.tensor(np_data, dtype=torch.float32)

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader

def calc_z_shapes(n_channel, input_size, n_block):
    z_shapes = []

    for i in range(n_block - 1):
        input_size //= 2
        n_channel *= 4

        z_shapes.append((n_channel, input_size, input_size, input_size))

    input_size //= 2
    z_shapes.append((n_channel * 8, input_size, input_size, input_size))

    return z_shapes

# modified by oosky
def calc_loss(log_p, logdet, image_size, n_bins, image_channel):
    # log_p = calc_log_p([z_list])
    n_pixel = image_size * image_size * image_size * image_channel

    loss = -log(n_bins) * n_pixel
    loss = loss + logdet + log_p

    return (
        (-loss / (log(2) * n_pixel)).mean(),
        (log_p / (log(2) * n_pixel)).mean(),
        (logdet / (log(2) * n_pixel)).mean(),
    )

def model_init(model, n_bins):
    init_dataset = sample_data(args.path, args.batch_init, "train", shuffle=True)
    init_loader = iter(init_dataset)
    init = next(init_loader)
    init = init.to(device)

    with torch.no_grad():
        log_p, logdet, _ = model.module(init + torch.rand_like(init) / n_bins)

    return model

def model_train(args, model, optimizer):

    writer = SummaryWriter()

    n_bins = 2. ** args.n_bits
    model = model_init(model, n_bins)

    best_loss = 99999

    train_dataset = sample_data(args.path, args.batch, "train", shuffle=True)
    valid_dataset = sample_data(args.path, args.batch, "valid", shuffle=False)

    z_sample = []
    z_shapes = calc_z_shapes(args.img_ch, args.img_size, args.n_block)
    for z in z_shapes:
        z_new = torch.randn(args.n_sample, *z) * args.temp
        z_sample.append(z_new.to(device))

    result = {}
    result["train/loss"] = []
    result["train/logp"] = []
    result["train/logdet"] = []

    result["valid/loss"] = []
    result["valid/logp"] = []
    result["valid/logdet"] = []

    for i in range(args.epochs):

        with tqdm(train_dataset) as tbar:
            print("TRAIN STEP: EPOCH {}".format(i+1))
            loss_t, logp_t, logdet_t = [], [], []

            for image in tbar:
                image = image.to(device)

                log_p, logdet, _ = model(image + torch.rand_like(image) / n_bins)

                logdet = logdet.mean()

                loss, log_p, log_det = calc_loss(log_p, logdet, args.img_size, n_bins, args.img_ch)
                model.zero_grad()
                loss.backward()
                optimizer.step()

                loss_t.append(loss.item())
                logp_t.append(log_p.item())
                logdet_t.append(log_det.item())

            result["train/loss"].append(statistics.mean(loss_t))
            result["train/logp"].append(statistics.mean(logp_t))
            result["train/logdet"].append(statistics.mean(logdet_t))

            writer.add_scalar('train/loss', result["train/loss"][-1], i+1)
            writer.add_scalar('train/logp', result["train/logp"][-1], i+1)
            writer.add_scalar('train/logdet', result["train/logdet"][-1], i+1)

            print(f'Loss: {result["train/loss"][-1]:.5f}; '
                  f'logp: {result["train/logp"][-1]:.5f}; '
                  f'logdet: {result["train/logdet"][-1]:.5f}')

            if (i + 1) % 10 == 0 or i == 0:
                print("VALID STEP: EPOCH {}".format(i+1))
                with torch.no_grad():
                    loss_v, logp_v, logdet_v = [], [], []
                    for image in valid_dataset:
                        image = image.to(device)

                        log_p, logdet, _ = model(image + torch.rand_like(image) / n_bins)

                        logdet = logdet.mean()

                        loss, log_p, log_det = calc_loss(log_p, logdet, args.img_size, n_bins, args.img_ch)

                        loss_v.append(loss.item())
                        logp_v.append(log_p.item())
                        logdet_v.append(log_det.item())

                    result["valid/loss"].append(statistics.mean(loss_v))
                    result["valid/logp"].append(statistics.mean(logp_v))
                    result["valid/logdet"].append(statistics.mean(logdet_v))

                    writer.add_scalar('valid/loss', result["valid/loss"][-1], i+1)
                    writer.add_scalar('valid/logp', result["valid/logp"][-1], i+1)
                    writer.add_scalar('valid/logdet', result["valid/logdet"][-1], i+1)

                    print(f'Loss: {result["valid/loss"][-1]:.5f}; '
                          f'logp: {result["valid/logp"][-1]:.5f}; '
                          f'logdet: {result["valid/logdet"][-1]:.5f}')

                    if best_loss > result["valid/loss"][-1]:
                        best_loss = result["valid/loss"][-1]

                        print("save model")
                        torch.save(
                            model.state_dict(), f'checkpoint/model_{str(i + 1).zfill(5)}.pt'
                        )
                        torch.save(
                            optimizer.state_dict(), f'checkpoint/optim_{str(i + 1).zfill(5)}.pt'
                        )

            if (i + 1) % 10 == 0 or i == 0:
                with torch.no_grad():
                    utils.save_image(
                        model_single.reverse(z_sample).cpu().data[:, :, 4, :, :],
                        f'sample/{str(i + 1).zfill(5)}.png',
                        normalize=False,
                        nrow=10,
                        range=(0.0, 1.0),
                    )

def model_test(args, model):

    test_dataset = sample_data(args.path, args.batch, "test", shuffle=False)
    model.eval()

    with tqdm(test_dataset) as tbar:
        oris, zs, xs = [], [], []

        for image in tbar:
            image = image.to(device)

            _, _, z = model(image)

            x = model_single.reverse(z)

            image = image.cpu().numpy()
            z = z.cpu().numpy()
            x = x.cpu().numpy()


            oris.append(image)
            zs.append(z)
            xs.append(x)

        ori = post(np.asarray(oris))
        z = post(np.asarray(zs))
        x = post(np.asarray(xs))

        np.save("./npy/ori.npy", ori)
        np.save("./npy/z.npy", z)
        np.save("./npy/x.npy", x)


def main(args, model):

    if args.mode == "train":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        model_train(args, model, optimizer)

    else:
        na = get_model_name(args.ckpt)
        model.load_state_dict(torch.load(na))
        model_test(args, model)



def arg_parser():
    parser = argparse.ArgumentParser(description='Glow trainer')
    parser.add_argument('--batch', default=256, type=int, help='batch size')
    parser.add_argument('--batch_init', default=512, type=int, help='initial batch size')
    parser.add_argument('--epochs', default=1000, type=int, help='maximum epochs')

    parser.add_argument(
        '--n_flow', default=8, type=int, help='number of flows in each block' # K
    )
    parser.add_argument('--n_block', default=2, type=int, help='number of blocks') # L
    parser.add_argument('--filter_size', default=64, type=int, help='number of filters')
    parser.add_argument(
        '--no_lu',
        action='store_true',
        help='use plain convolution instead of LU decomposed version',
    )
    parser.add_argument(
        '--affine', action='store_true', help='use affine coupling instead of additive'
    )
    parser.add_argument('--n_bits', default=5, type=int, help='number of bits') # float32 -> 5
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--img_size', default=8, type=int, help='image size')
    parser.add_argument('--img_ch', default=1, type=int, help='image channel') # RGB -> 3, Gray -> 1
    parser.add_argument('--temp', default=0.7, type=float, help='temperature of sampling')
    parser.add_argument('--n_sample', default=100, type=int, help='number of samples')
    parser.add_argument('--path', default='./np_record2/', type=str, help='Path to image directory')
    parser.add_argument('--ckpt', default='./checkpoint/', type=str, help='Path to checkpoint directory')
    parser.add_argument('--mode', default='train', type=str, help='[train, test]')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = arg_parser()

    model_single = Glow(
        in_channel=args.img_ch,
        n_flow=args.n_flow,
        n_block=args.n_block,
        filter_size=args.filter_size,
        affine= not args.affine,
        conv_lu= not args.no_lu
    )

    model = nn.DataParallel(model_single)
    model = model.to(device)

    main(args, model)

