import argparse
import os
import torch


class BaseOptions():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False
        self.message = None


    def initialize(self, parser):
        parser.add_argument('--dataroot',default='/data/yujingbo/repos/dataset/CelebA_stylegan.db',required=False,type=str, help='path of specified dataset')
        parser.add_argument(
            '--phase',
            type=int,
            default=600_000,
            help='number of samples used for each training phases',
        )
        parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
        parser.add_argument('--sched', action='store_true', help='use lr scheduling')
        parser.add_argument('--init_size', default=8, type=int, help='initial image size')
        parser.add_argument('--max_size', default=1024, type=int, help='max image size')
        parser.add_argument('--gpu_ids', type=str, default='1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument(
            '--no_from_rgb_activate',
            action='store_true',
            help='use activate in from_rgb (original implementation)',
        )
        parser.add_argument(
            '--mixing', action='store_true', help='use mixing regularization'
        )
        parser.add_argument(
            '--loss',
            type=str,
            default='wgan-gp',
            choices=['wgan-gp', 'r1'],
            help='class of gan loss',
        )
        parser.add_argument('--load_time',required=False, type=str, help='指出具体使用哪个模型，一般用时间表示，故而是time')
        parser.add_argument('--isTrain', required=False,default=1,type=int , help='标识是否在训练，从而决定模型一些属性的设置')
        parser.add_argument('--batch_size', required=False,default=1,type=int , help='batch size')
        parser.add_argument('--checkpoints_dir', required=False,default='/data/yujingbo/repos/checkpoints_of_all_repos',type=str , help='模型存放的位置，这个一般不变')
        parser.add_argument('--model_name', required=False,default='My_Style_GAN',type=str , help='使用的模型的名字')



        # ---------------------------------------------------- 使用pytorch跑 cycle stylegan---------------------------------------------------------
        parser.add_argument(
            "--size", type=int, default=1024, help="output image size of the generator"
        )
        parser.add_argument(
            "--sample",
            type=int,
            default=1,
            help="number of samples to be generated for each image",
        )
        parser.add_argument(
            "--pics", type=int, default=20, help="number of images to be generated"
        )
        parser.add_argument("--truncation", type=float, default=1, help="truncation ratio")
        parser.add_argument(
            "--truncation_mean",
            type=int,
            default=4096,
            help="number of vectors to calculate mean for the truncation",
        )
        parser.add_argument(
            "--ckpt",
            type=str,
            default="stylegan2-ffhq-config-f.pt",
            help="path to the model checkpoint",
        )
        parser.add_argument(
            "--channel_multiplier",
            type=int,
            default=2,
            help="channel multiplier of the generator. config-f = 2, else = 1",
        )

        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # modify model-related parser options
        #model_name = opt.model
        #model_option_setter = models.get_option_setter(model_name)
        #parser = model_option_setter(parser, self.isTrain)
        #opt, _ = parser.parse_known_args()  # parse again with new defaults

        # modify dataset-related parser options
        #dataset_name = opt.dataset_mode
        #dataset_option_setter = data.get_option_setter(dataset_name)
        #parser = dataset_option_setter(parser, self.isTrain)

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)
        self.message = message

        # save to the disk
        '''
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')
        '''

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        #opt.isTrain = self.isTrain   # train or test

        # process opt.suffix
        #if opt.suffix:
        #    suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
        #    opt.name = opt.name + suffix

        self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt,self.message
