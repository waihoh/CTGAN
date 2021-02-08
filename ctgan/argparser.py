import argparse
import os
from ctgan import config as cfg
import pandas as pd
import numpy as np
from ctgan.transformer import DataTransformer

# To allow True/False argparse input.
# See answer by Maxim in https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def _parse_args():
    parser = argparse.ArgumentParser(description='Command Line Interface')
    parser.add_argument("-nv", "--nv", help="run nvidia-smi command", action="store_true")
    parser.add_argument("--torch_seed", default=0, type=int, metavar='', help="PyTorch random seed")
    parser.add_argument("--numpy_seed", default=0, type=int, metavar='', help="PyTorch random seed")
    parser.add_argument('--model', default=None, type=str, metavar='', help='ctgan, tablegan or tvae')
    parser.add_argument('--datadir', default="/workspace", type=str, metavar='', help='path of training data directory')
    parser.add_argument('--outputdir', default="/workspace", type=str, metavar='', help='path of output directory')
    parser.add_argument('--data_fn', default=None, type=str, metavar='', help='filename of transformed training data (with .csv)')
    parser.add_argument('--val_data_fn', default=None, type=str, metavar='',help='filename of validation data (with .csv)')
    parser.add_argument('--threshold', default=None, type=str, metavar='',help='threshold of KLD (with .csv)')
    parser.add_argument('--transformer', default=None, type=str, metavar='',help='VGM Transformer')
    parser.add_argument('--discrete_fn', default=None, type=str, metavar='', help='filename of discrete cols, (with .txt)')
    parser.add_argument('--samplesize', default=None, type=int, metavar='', help='synthetic sample size')
    parser.add_argument('--trials', default=10, type=int, metavar='', help='Number of Optuna trials')

    # CTGAN parameters
    parser.add_argument('--ct_embedding', default=None, type=int, metavar='', help='ctgan embedding')
    parser.add_argument('--ct_depth', default=None, type=int, metavar='', help='ctgan num hidden layers')
    parser.add_argument('--ct_width', default=None, type=int, metavar='', help='ctgan width of mlp')
    parser.add_argument('--ct_gen_lr', default=None, type=float, metavar='', help='ctgan generator learning rate')
    parser.add_argument('--ct_dis_lr', default=None, type=float, metavar='', help='ctgan discriminator learning rate')
    parser.add_argument('--ct_batchsize', default=None, type=int, metavar='', help='ctgan batch size')
    parser.add_argument('--ct_epochs', default=None, type=int, metavar='', help='ctgan num epochs')
    parser.add_argument('--ct_dropout', default=None, type=float, metavar='', help='ctgan dropout rate')
    parser.add_argument('--ct_dis_step', default=None, type=int, metavar='', help='ctgan discriminator step')
    parser.add_argument('--ct_device', default=None, type=str, metavar='', help='ctgan cpu or cuda')

    # TableGAN parameters
    parser.add_argument('--tbl_embedding', default=None, type=int, metavar='', help='tablegan embedding')
    parser.add_argument('--tbl_num_channels', default=None, type=int, metavar='', help='tablegan num channels')
    parser.add_argument('--tbl_dlayer', default=None, type=int, metavar='', help='tablegan: 0 default, -1 to remove one layer, 1 to add a 1X1 conv')
    parser.add_argument('--tbl_lr', default=None, type=float, metavar='', help='tablegan learning rate')
    parser.add_argument('--tbl_batchsize', default=None, type=int, metavar='', help='tablegan batch size')
    parser.add_argument('--tbl_epochs', default=None, type=int, metavar='', help='tablegan num epochs')
    parser.add_argument('--tbl_dis_step', default=None, type=int, metavar='', help='tablegan discriminator step')
    parser.add_argument('--tbl_device', default=None, type=str, metavar='', help='tablegan cpu or cuda')

    # TVAE parameters
    parser.add_argument('--tv_embedding', default=None, type=int, metavar='', help='tvae embedding')
    parser.add_argument('--tv_condgen', default=None, type=str2bool, metavar='', help='tvae cond. gen.')
    parser.add_argument('--tv_depth', default=None, type=int, metavar='', help='tvae num hidden layers')
    parser.add_argument('--tv_width', default=None, type=int, metavar='', help='tvae width of mlp')
    parser.add_argument('--tv_lr', default=None, type=float, metavar='', help='tvae learning rate')
    parser.add_argument('--tv_batchsize', default=None, type=int, metavar='', help='tvae batch size')
    parser.add_argument('--tv_epochs', default=None, type=int, metavar='', help='tvae num epochs')
    parser.add_argument('--tv_device', default=None, type=str, metavar='', help='tvae cpu or cuda')

    return parser.parse_args()


class ParserOutput:
    """
        To store output values in an object instead of returning individual values
        The default datadir and outputdir are set to /workspace for ease of using Docker container.
    """
    def __init__(self):
        self.proceed = False
        self.torch_seed = 0
        self.numpy_seed = 0
        self.model_type = None
        self.datadir = None
        self.outputdir = None
        self.data_fn = None
        self.discrete_fn = None
        self.val_data_fn = None
        self.threshold = None
        self.transformer = None
        self.samplesize = 9905  # current size of test data.
        self.trials = None

        self.parser_func()

    def parser_func(self):
        """
        The function acts as a placeholder to update cfg with valeus from argparse.

        Returns:
            proceed: continue with subsequent code in main.oy
            model: the selected model is either ctgan, tablegan or tvae.
            datadir: where the training data is located.
            outputdir: where the trained model should be stored.
            data_fn: file name of training data.
            discrete_fn: file that contains the names of discrete variables.
        """

        args = _parse_args()

        # enable user to run the nvidia-smi command
        # without having to run Docker container in interactive mode.
        if args.nv:
            os.system("nvidia-smi")
            return

        # sanity check
        if args.model is None:
            print('Please specify --model.')
            return

        if args.data_fn is None:
            print('Please specify --data_fn.')
            return

        if args.discrete_fn is None:
            print('Please specify --discrete_fn')
            return

        # Store values
        self.torch_seed = args.torch_seed
        self.numpy_seed = args.numpy_seed
        self.model_type = args.model.lower()
        self.datadir = args.datadir
        self.outputdir = args.outputdir
        self.data_fn = args.data_fn
        self.val_data_fn = args.val_data_fn
        self.discrete_fn = args.discrete_fn
        self.trials = args.trials

        if args.val_data_fn is not None:
            self.val_data_fn = pd.read_csv(os.path.join(self.datadir, args.val_data_fn))

        if args.threshold is not None:
            self.threshold = np.transpose(pd.read_csv(os.path.join(self.datadir, args.threshold)))

        if args.transformer is not None:
            transformer_path = os.path.join(self.datadir, args.transformer)
            self.transformer = DataTransformer.load(transformer_path)

        if args.samplesize is not None:
            self.samplesize = args.samplesize

        if self.model_type == 'ctgan':
            if args.ct_embedding is not None:
                cfg.ctgan_setting.EMBEDDING = args.ct_embedding

            if args.ct_depth is not None:
                cfg.ctgan_setting.DEPTH = args.ct_depth

            if args.ct_width is not None:
                cfg.ctgan_setting.WIDTH = args.ct_width

            if args.ct_gen_lr is not None:
                cfg.ctgan_setting.GENERATOR_LEARNING_RATE = args.ct_gen_lr

            if args.ct_dis_lr is not None:
                cfg.ctgan_setting.DISCRIMINATOR_LEARNING_RATE = args.ct_dis_lr

            if args.ct_batchsize is not None:
                cfg.ctgan_setting.BATCH_SIZE = args.ct_batchsize

            if args.ct_epochs is not None:
                cfg.ctgan_setting.EPOCHS = args.ct_epochs

            if args.ct_dropout is not None:
                cfg.ctgan_setting.DROPOUT = args.ct_dropout

            if args.ct_dis_step is not None:
                cfg.ctgan_setting.DISCRIMINATOR_STEP = args.ct_dis_step

            if args.ct_device is not None:
                cfg.ctgan_setting.DEVICE = args.ct_device

        elif self.model_type == 'tablegan':

            if args.tbl_embedding is not None:
                cfg.tablegan_setting.EMBEDDING = args.tbl_embedding

            if args.tbl_num_channels is not None:
                cfg.tablegan_setting.NUM_CHANNELS = args.tbl_num_channels

            if args.tbl_dlayer is not None:
                cfg.tablegan_setting.DLAYER = args.tbl_dlayer

            if args.tbl_lr is not None:
                cfg.tablegan_setting.LEARNING_RATE = args.tbl_lr

            if args.tbl_batchsize is not None:
                cfg.tablegan_setting.BATCH_SIZE = args.tbl_batchsize

            if args.tbl_epochs is not None:
                cfg.tablegan_setting.EPOCHS = args.tbl_epochs

            if args.tbl_dis_step is not None:
                cfg.tablegan_setting.DISCRIMINATOR_STEP = args.tbl_dis_step

            if args.tbl_device is not None:
                cfg.tablegan_setting.DEVICE = args.tbl_device

        elif self.model_type == 'tvae':
            if args.tv_embedding is not None:
                cfg.tvae_setting.EMBEDDING = args.tv_embedding

            if args.tv_condgen is not None:
                cfg.tvae_setting.CONDGEN = args.tv_condgen

            if args.tv_depth is not None:
                cfg.tvae_setting.DEPTH = args.tv_depth

            if args.tv_width is not None:
                cfg.tvae_setting.WIDTH = args.tv_width

            if args.tv_lr is not None:
                cfg.tvae_setting.LEARNING_RATE = args.tv_lr

            if args.tv_batchsize is not None:
                cfg.tvae_setting.BATCH_SIZE = args.tv_batchsize

            if args.tv_epochs is not None:
                cfg.tvae_setting.EPOCHS = args.tv_epochs

            if args.tv_device is not None:
                cfg.tvae_setting.DEVICE = args.tv_device
        else:
            print('Please specify the correct model type.')
            return

        self.proceed = True
