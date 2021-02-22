# centralized configuration file.
# It can be updated using argparse as well.



class ctgan_setting:
    EMBEDDING = 128
    DEPTH = 2 ## or 3 hidden layers
    WIDTH = 256
    GENERATOR_LEARNING_RATE = 2e-4 ## or 2e-3;2e-5
    DISCRIMINATOR_LEARNING_RATE = 2e-4 ## or 2e-3;2e-5

    BATCH_SIZE = 500 ## or 1000
    EPOCHS = 300 ## or 600
    DROPOUT = 0.5 ## or 0.25
    DISCRIMINATOR_STEP = 1
    DEVICE = "cpu"  # "cuda:0"


class tvae_setting:
    EMBEDDING = 128
    CONDGEN = True
    DEPTH = 2 ## or 3 hidden layers
    WIDTH = 128
    LEARNING_RATE = 1e-3 ## 1e-2; 1e-4
    BATCH_SIZE = 500 ## or 1000
    EPOCHS = 300  ## or 600
    DEVICE = "cpu"  # "cuda:0"


class tablegan_setting:
    EMBEDDING = 100
    NUM_CHANNELS = 64
    DLAYER = 0 # 0: no changes. -1: remove last item in layer_dims, 1: add a 1X1 Convolution layer.
    STRIDE = 2  # This is fixed
    KERNEL_SIZE = 4  # This is fixed
    SCALE_FACTOR = 2  # This is fixed
    LEARNING_RATE = 2e-4 ## 2e-3;2e-5
    BATCH_SIZE = 500 ## or 1000
    EPOCHS = 300  ## or 600
    DISCRIMINATOR_STEP = 1
    DEVICE = "cpu"  # "cuda:0"


# # split to different objects. easier to manage and update different models separately.
# CTGAN = ctgan_setting()
# TVAE = tvae_setting()
# TABLEGAN = tablegan_setting()
#
# # Common parameters across all models.
# #OPTIMIZER = "ADAM"
