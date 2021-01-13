import argparse
import os
from ctgan import config as cfg

def _parse_args():
    parser = argparse.ArgumentParser(description='Command Line Interface')
    parser.add_argument('--model', default=None, type=str, metavar='', help='ctgan, tablegan or tvae')
    parser.add_argument('--datadir', default=os.getcwd(), type=str, metavar='', help='path of training data directory')
    parser.add_argument('--outputdir', default=None, type=str, metavar='', help='path of output directory')
    parser.add_argument('--data_fn', default=os.getcwd(), type=str, metavar='', help='filename of training data (with .csv)')
    parser.add_argument('--discreet_fn', default=None, type=str, metavar='', help='filename of discreet cols, (with .txt)')

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
    parser.add_argument('--tv_depth', default=None, type=int, metavar='', help='tvae num hidden layers')
    parser.add_argument('--tv_width', default=None, type=int, metavar='', help='tvae width of mlp')
    parser.add_argument('--tv_lr', default=None, type=float, metavar='', help='tvae learning rate')
    parser.add_argument('--tv_batchsize', default=None, type=int, metavar='', help='tvae batch size')
    parser.add_argument('--tv_epochs', default=None, type=int, metavar='', help='tvae num epochs')
    parser.add_argument('--tv_device', default=None, type=str, metavar='', help='tvae cpu or cuda')

    return parser.parse_args()

def parser_func():
    '''
    The function acts as a placeholder to update cfg with valeus from argparse.

    Returns:
        model: the selected model is either ctgan, tablegan or tvae.
        datadir: where the training data is located.
        outputdir: where the trained model should be stored.
        data_fn: file name of training data.
        discreet_fn: file that contains the names of discreet variables.
    '''

    args = _parse_args()
    model_type = args.model.lower()
    datadir = args.datadir
    outputdir = args.outputdir
    data_fn = args.data_fn
    discreet_fn = args.discreet_fn


    if model_type == 'ctgan':
        if args.ct_embedding is not None:
            cfg.ctgan_setting.EMBEDDING = args.ct_embedding

        if args.ct_depth is not None:
            cfg.ctgan_setting.DEPTH = args.ct_depth

        if args.ct_width is not None:
            cfg.ctgan_setting.WIDTH = args.ct_width

        if args.ct_gen_lr is not None:
            cfg.ctgan_setting.GERENATOR_LEARNING_RATE = args.ct_gen_lr

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

    elif model_type == 'tablegan':

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

    elif model_type == 'tvae':
        if args.tv_embedding is not None:
            cfg.tvae_setting.EMBEDDING = args.tv_embedding

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

    return model_type, datadir, outputdir, data_fn, discreet_fn

