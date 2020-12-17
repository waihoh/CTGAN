# centralized configuration file.
# It can be updated using argparse as well.

class ctgan_setting:
    EMBEDDING = 128
    DEPTH = 2 ## or 3 hidden layers
    WIDTH = 256
    LEARNING_RATE = 2e-4 ## or 2e-3;2e-5
    BATCH_SIZE = 500 ## or 1000
    EPOCHS = 300 ## or 600
    DROPOUT = 0.5 ## or 0.25

###tvae and tablegan are not finished

class tvae_setting:
    EMBEDDING = 128
    DEPTH = 2
    WIDTH = 8
    LEARNING_RATE = 1e-3 ## 1e-2; 1e-4
    BATCH_SIZE = 500 ## or 1000
    EPOCHS = 300  ## or 600


class tablegan_setting:
    NUM_FEATURES = 60
    FILTER_WIDTH = 3
    LEARNING_RATE = 2e-4 ## 2e-3;2e-5
    BATCH_SIZE = 500 ## or 1000
    EPOCHS = 300  ## or 600


# split to different objects. easier to manage and update different models separately.
CTGAN = ctgan_setting()
TVAE = tvae_setting()
TABLEGAN = tablegan_setting()

# Common parameters across all models.
#OPTIMIZER = "ADAM"
#ctgan_depth = 999



