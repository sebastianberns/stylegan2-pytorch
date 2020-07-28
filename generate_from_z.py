import argparse

import torch
from torchvision import utils
from model import Generator
from pathlib import Path
from tqdm import tqdm


def generate(args, g_ema):
    with torch.no_grad():
        g_ema.eval()

        # if args.truncation < 1:
        #     mean_latent = g_ema.mean_latent(args.truncation_mean)
        # else:
        #     mean_latent = None
        mean_latent = g_ema.mean_latent(args.truncation_mean)

        for i in tqdm(range(len(args.samples_z))):
            sample_z = torch.load(args.samples_z[i], map_location=args.device)
            sample, _ = g_ema([sample_z], truncation=args.truncation, truncation_latent=mean_latent)

            # Create directory if it does not exist
            dir = Path(args.savedir)
            dir.mkdir(parents=True, exist_ok=True)
            name = str(i).zfill(6)
            utils.save_image(
                sample,
                dir/f'{name}.png',
                nrow=1,
                normalize=True,
                range=(-1, 1),
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--truncation', type=float, default=1)
    parser.add_argument('--truncation_mean', type=int, default=4096)
    parser.add_argument('--ckpt', type=str, default="stylegan2-ffhq-config-f.pt")
    parser.add_argument('--channel_multiplier', type=int, default=2)
    parser.add_argument('--savedir', type=str, default="sample/")
    parser.add_argument('samples_z', nargs='+')

    args = parser.parse_args()

    args.latent = 512
    args.n_mlp = 8

    g_ema = Generator(
        args.size,
        args.latent,
        args.n_mlp,
        channel_multiplier=args.channel_multiplier
    ).to(args.device)

    checkpoint = torch.load(args.ckpt)
    g_ema.load_state_dict(checkpoint['g_ema'])

    generate(args, g_ema)
