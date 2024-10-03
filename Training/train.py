import argparse
import os
import time
from random import shuffle

from datasets import GRAB_DATA, MINI_BATCH, get_data_paths
from model import ISIZE, NC_IN, NC_OUT, NET_D_TRAIN, NET_G, NET_G_TRAIN

# parse the optional arguments:
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", help="name of model", default="test_1")

parser.add_argument(
    "--display_iter",
    help="number of iterations between each test",
    type=int,
    default=20000,
)
parser.add_argument(
    "--max_iter", help="total number of iterations", type=int, default=500000
)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--input", help="Input SQL data", default="aia.np_path_normal")
parser.add_argument("--output", help="Output SQL data", default="hmi.np_path_normal")
parser.add_argument(
    "--connector",
    nargs="+",
    help="SQL connection between input and output",
    default=["aia.id", "hmi.aia_id"],
)
parser.add_argument(
    "--tol", help="tolerance on image time difference in hours", type=float, default=3
)
parser.add_argument("--kernel", help="kernel size", type=int, default=4)
parser.add_argument("--mask", help="use mask on generator output", action="store_true")

args = parser.parse_args()

# Hyper parameters
# number of iterations before display and model creation
DISPLAY_ITERS = args.display_iter
NITERS = args.max_iter  # total number of iterations
BATCH_SIZE = args.batch_size  # number of images in each batch
TRIAL_NAME = args.model_name

# make a folder for the trial if it doesn't already exist
MODEL_PATH = "./Models/" + TRIAL_NAME + "/"
os.makedirs(MODEL_PATH) if not os.path.exists(MODEL_PATH) else None

data_paths = get_data_paths("./image.db", args.input, args.output, args.connector)
LIST_TOTAL = shuffle(list(GRAB_DATA(data_paths, args.tol)))
assert len(LIST_TOTAL) > 0, "No Images satisfy constraints (check --tol)"
print(f"Training on {len(LIST_TOTAL)} images.")

# creates a generator to use for training
TRAIN_BATCH = MINI_BATCH(LIST_TOTAL, BATCH_SIZE, NC_IN, NC_OUT)

# initialise training variables
T0 = T1 = time.time()
GEN_ITERS = 0
ERR_L = 0
EPOCH = 0
ERR_G = 0
ERR_L_SUM = 0
ERR_G_SUM = 0
ERR_D_SUM = 0

# training:
i = 0
while GEN_ITERS <= NITERS:
    EPOCH, TRAIN_A, TRAIN_B = next(TRAIN_BATCH)
    # input data set
    TRAIN_A = TRAIN_A.reshape((BATCH_SIZE, ISIZE, ISIZE, NC_IN))
    # output data set
    TRAIN_B = TRAIN_B.reshape((BATCH_SIZE, ISIZE, ISIZE, NC_OUT))

    # descriminator training and error
    (ERR_D,) = NET_D_TRAIN([TRAIN_A, TRAIN_B])
    ERR_D_SUM += ERR_D

    # generator training and error
    ERR_G, ERR_L = NET_G_TRAIN([TRAIN_A, TRAIN_B])
    ERR_G_SUM += ERR_G
    ERR_L_SUM += ERR_L

    GEN_ITERS += 1

    f = open(MODEL_PATH + "iter_loss.txt", "a+")
    update_str = (
        f"[ {EPOCH:0>3d} ][ {GEN_ITERS:0>6d} / {NITERS} ] LOSS_D: "
        f"{ERR_D:0>5.3f} LOSS_G: {ERR_G:0>5.3f} LOSS_L: {ERR_L:0>5.3f}\n"
    )
    f.write(update_str)
    f.close()

    # print training summary and save model
    if GEN_ITERS % DISPLAY_ITERS == 0:
        print(
            "[%d][%d/%d] LOSS_D: %5.3f LOSS_G: %5.3f LOSS_L: %5.3f T:"
            "%dsec/%dits, Total T: %d"
            % (
                EPOCH,
                GEN_ITERS,
                NITERS,
                ERR_D_SUM / DISPLAY_ITERS,
                ERR_G_SUM / DISPLAY_ITERS,
                ERR_L_SUM / DISPLAY_ITERS,
                time.time() - T1,
                DISPLAY_ITERS,
                time.time() - T0,
            )
        )

        ERR_L_SUM = 0
        ERR_G_SUM = 0
        ERR_D_SUM = 0
        DST_MODEL = MODEL_PATH + "ITER" + "%07d" % GEN_ITERS + ".h5"
        NET_G.save(DST_MODEL)
        T1 = time.time()

    i += 1
