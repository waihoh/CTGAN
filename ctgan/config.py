# centralized configuration file.
# It can be updated using argparse as well.



class ctgan_setting:
    EMBEDDING = 128
    DEPTH = 2
    WIDTH = 256
    GENERATOR_LEARNING_RATE = 2e-4
    DISCRIMINATOR_LEARNING_RATE = 2e-4

    BATCH_SIZE = 500
    EPOCHS = 300
    DROPOUT = 0.5
    DISCRIMINATOR_STEP = 1
    DEVICE = "cpu"  # "cuda:0"


class tvae_setting:
    EMBEDDING = 128
    DEPTH = 2
    WIDTH = 128
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 500
    EPOCHS = 300
    CONDGEN_ENCODER = True
    CONDGEN_LATENT = True
    OPTUNA_ELBO = False  # Use ELBO as Optuna metric
    DEVICE = "cpu"  # "cuda:0"


class tablegan_setting:
    EMBEDDING = 100
    NUM_CHANNELS = 64
    DLAYER = 0 # 0: no changes. -1: remove last item in layer_dims, 1: add a 1X1 Convolution layer.
    STRIDE = 2  # This is fixed
    KERNEL_SIZE = 4  # This is fixed
    SCALE_FACTOR = 2  # This is fixed
    LEARNING_RATE = 2e-4
    BATCH_SIZE = 500
    EPOCHS = 300
    DISCRIMINATOR_STEP = 1
    DEVICE = "cpu"  # "cuda:0"
