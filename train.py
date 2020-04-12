import os
import argparse
import yaml
import logging
import torch
import torch.backends.cudnn as cudnn
# from apex import amp
from utils.logging import open_log
from utils.tools import load_checkpoint
from models import ClsNet
# from torchsummaryX import summary


def arg_parse():
    parser = argparse.ArgumentParser(
        description='ClsNet')
    parser.add_argument('-cfg', '--config', default='configs/resnet-attention.yaml',
                        type=str, help='load the config file')
    parser.add_argument('--cuda', default=True,
                        type=bool, help='Use cuda to train model')
    args = parser.parse_args()
    return args


def main():
    args = arg_parse()
    config = yaml.load(open(args.config))

    gpus = ','.join([str(i) for i in config['GPUs']])
    os.environ["CUDA_VISIBLE_DEVICES"]=gpus

    # open log file
    open_log(args)
    logging.info(args)
    logging.info(config)

    logging.info(config['Data_CLASSES'])
    logging.info('Using the network: {}'.format(config['arch']))

    # Set net
    ClsModel = ClsNet.build_model(config)

    if args.cuda:
        ClsModel.cuda()
        cudnn.benchmark = True

    # Show the net architecture
    # summary(ClsModel, torch.zeros(1, 1, 269, 512, 512).cuda())

    if config['Using_pretrained_weights']:
        ClsModel.load_pretrained_weights()
    if config['Cls']['resume'] != None:
        load_checkpoint(ClsModel, config['Cls']['resume'])

    from utils import net_utils
    optimizer, train_loader, valid_loader = net_utils.prepare_net(config, ClsModel)

    if config['Using_apex']:
        amp.register_float_function(torch, 'sigmoid')
        amp.initialize(ClsModel, optimizer, opt_level='O1', max_loss_scale=2 ** 20)

    ClsModel = torch.nn.DataParallel(ClsModel)

    net_utils.train_net(optimizer, train_loader, valid_loader, ClsModel, config)


if __name__ == '__main__':
    main()
