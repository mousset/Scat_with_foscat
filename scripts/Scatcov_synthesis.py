import numpy as np
import sys
import healpy as hp
import argparse
import os

import foscat.Synthesis as synthe
import foscat.scat_cov as sc
import almscat.Sphere_lib as sphlib

import tensorflow as tf
print('Num GPUs Available:', len(tf.config.experimental.list_physical_devices('GPU')))
print('Available devices:', tf.config.list_physical_devices())


########## ARGUMENTS
parser = argparse.ArgumentParser()

parser.add_argument('-n', '--nside', help='NSIDE value', default=32, type=int)
parser.add_argument('-e', '--epochs', help='Number of epochs', default=10, type=int)
parser.add_argument('-a', '--all_type', help='Type float32 or float64.', default='float32', type=str)
parser.add_argument('-c', '--cond_ini', help='Initial condition type', default='white', type=str)
parser.add_argument('-s', '--save_dir', help='Path where outputs are saved.', default='./', type=str)

args = parser.parse_args()

# If save_dir does not exist we create the directory
if not os.path.exists(args.save_dir):
   os.makedirs(args.save_dir)

nside = args.nside  # NSIDE

########## DEFAULT PARAMETERS
use_R_format = True
norm = 'auto'  # Normalisation
KERNELSZ = 5
if KERNELSZ == 5:
    LAMBDA = 1.
elif KERNELSZ == 3:
    LAMBDA = 1.2
else:
    raise ValueError('KERNELSZ must be 3 or 5.')

########## LOSS FUNCTIONS
def lossP00(x, scat_op, args):

    tP00 = args[0]
    norm = args[1]

    # Compute P00 on the iterating image
    xP00 = scat_op.eval(x, image2=None, norm=norm).P00

    loss = scat_op.reduce_sum(scat_op.square(xP00 - tP00))

    return loss


def lossAll(x, scat_op, args):

    tcoeff = args[0]
    norm = args[1]

    # Compute P00 on the iterating image
    xcoeff = scat_op.eval(x, image2=None, norm=norm)

    loss = scat_op.reduce_sum(scat_op.square(xcoeff - tcoeff))
    return loss

########## RUN THE SYNTHESIS


if __name__ == "__main__":
    ##### TARGET MAP
    tmap = sphlib.make_hpx_planet(nside=nside, planet='venus', interp=True, normalize=True, nest=True)  # Venus

    ##### Initialize the operator
    scat_op = sc.funct(NORIENT=3,   # define the number of wavelet orientation
                       KERNELSZ=KERNELSZ,  # define the kernel size (here 5x5)
                       OSTEP=-1,     # get very large scale (nside=1)
                       LAMBDA=LAMBDA,
                       TEMPLATE_PATH='../data',
                       slope=1.,
                       use_R_format=use_R_format,
                       all_type=args.all_type)

    ##### Coeffs of the target image
    scat_op.clean_norm()
    tcoeff = scat_op.eval(tmap, image2=None, norm=norm)
    tP00 = tcoeff.P00

    ##### Choose the loss
    loss = synthe.Loss(lossAll, scat_op, tcoeff, norm)
    # loss = synthe.Loss(lossP00, scat_op, tP00, norm)

    ##### Build the synthesis class
    sy = synthe.Synthesis([loss])

    ##### Initial condition MAP
    # Fix the seed
    np.random.seed(42)

    if args.cond_ini == 'white':
        # Make white gaussian noise
        imap = np.random.randn(12*nside*nside)
    elif args.cond_ini == 'colour':
        # Gaussian noise with the target PS
        cl = hp.anafast(hp.reorder(tmap, n2r=True))
        imap = hp.reorder(hp.synfast(cl, nside), r2n=True)
    elif args.cond_ini == 'noisy_target':
        imap = tmap + np.random.randn(12 * nside * nside)
    else:
        raise ValueError('Initial condition not defined.')

    ##### Run the synthesis
    omap = sy.run(imap,
                  DECAY_RATE=0.9998,
                  NUM_EPOCHS=args.epochs,
                  LEARNING_RATE=0.03,
                  EPSILON=1e-16)

    ##### Compute the coeff
    icoeff = scat_op.eval(imap, image2=None, norm=norm)
    ocoeff = scat_op.eval(omap, image2=None, norm=norm)

    ##### Store the results
    np.save(args.save_dir + 'target_map.npy', tmap)
    np.save(args.save_dir + 'initial_map.npy', imap)
    np.save(args.save_dir + 'output_map.npy', omap)
    np.save(args.save_dir + 'loss.npy', sy.get_history())

    tcoeff.save(args.save_dir + 'target_coeff')
    icoeff.save(args.save_dir + 'initial_coeff')
    ocoeff.save(args.save_dir + 'output_coeff')

    print('Computation Done')
    sys.stdout.flush()