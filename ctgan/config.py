# centralized configuration file.
# It can be updated using argparse as well.

class ctgan_setting:
    DEPTH = 10
    WIDTH = 5


class tvae_setting:
    EMBEDDING = 128
    DEPTH = 20
    WIDTH = 8


class tablegan_setting:
    NUM_FEATURES = 60
    FILTER_WIDTH = 3


# split to different objects. easier to manage and update different models separately.
CTGAN = ctgan_setting()
TVAE = tvae_setting()
TABLEGAN = tablegan_setting()

# Common parameters across all models.
LEARNING_RATE = 0.001
OPTIMIZER = "ADAM"


