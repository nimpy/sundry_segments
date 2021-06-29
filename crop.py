import argparse
import numpy as np
import imageio


parser = argparse.ArgumentParser()
parser.add_argument('input', default='image.png',
                    help="Path to the input image to be cropped")
parser.add_argument('--output',
                    help="Path to the output image to be saved")

parser.add_argument('--h1', default='0',
                    help="Height coordinate begin")
parser.add_argument('--w1', default='0',
                    help="Width coordinate begin")

parser.add_argument('--h2', default='10',
                    help="Height coordinate end")
parser.add_argument('--w2', default='20',
                    help="Width coordinate end")

parser.add_argument('--rgb', default='1',
                    help="Is the image RGB? (1 for yes, 0 for no)")


args = parser.parse_args()

image = imageio.imread(args.input)

if args.output is None:
    # args.output = args.input[: args.input.rfind(.) - 1] + '_crop' + args.input[args.input.rfind(.) - 1 :]
    args.output = args.input[: args.input.rfind('.')] + '_crop' + args.input[args.input.rfind('.') :]



h1 = int(args.h1)
w1 = int(args.w1)
h2 = int(args.h2)
w2 = int(args.w2)
rgb = int(args.rgb)

if rgb:
    image_crop = image[h1:h2, w1:w2, :]
else:
    image_crop = image[h1:h2, w1:w2]

imageio.imsave(args.output, image_crop)
