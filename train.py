
class Train_wrapper:
    def __init__(self,config_class,config_num):
        self.config_class = config_class
        self.config_num = config_num
        self.args = self.init_args()
        self.device = self.args.device

        getattr(self,f"init_config_{self.config_class}")()

    def init_args(self):
        # parser = argparse.ArgumentParser(description="trainer")
        # parser.add_argument(
        #     "--example", type=int, default=0, help="example"
        # )
        # opt = parser.parse_args()

        cla = getattr(config, f"Config_{self.config_class}")
        args = cla().get_config(config_num)
        args.device = f"cuda"
        home = os.environ['HOME']
        args.name = "example"

        # for visualize
        if 1:
            args.use_html = False  # 是否将训练中的图片存成html
            args.display_id = 1  # 大于0才会使用visdom
            args.display_winsize = 500  #
            args.display_port = 8097  # 端口，一般用默认的8097
            args.display_ncols = 3  # 展示训练中生成的图片，决定每行有几个图片
            args.display_server = '127.0.0.1'  # 一般 127.0.0.1
            args.display_env = args.name  # visdom的展示网页中有不同的子环境，一般根据名字设定环境，不会混
            args.web_dir = os.path.join(home, 'repos/web_dir/', args.name)  # 统一存放html的文件夹
            args.display_loss = True  # 是否使用tensorboardx展示loss
            args.print_in_stdout = True  # 是否将损失等信息输出到命令行
            args.use_tensorboard = True
            args.tensorboard_dir = os.path.join(home, 'repos/runs/', args.name)
            args.header = args.name

        return args

    def init_config_a(self):
        return

    def init_config_b(self):
        return

    def start(self):

        trainer = getattr(Trainer(wrapper),f"train_config_{self.config_class}")
        print("starting training ...")
        trainer()

        return


if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    import os,argparse,config

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    config_class = "a"
    config_num = "1"  # num | test | pretrain_d | pretrain_g | new_d_1 | pre_classify_1

    from coach import Trainer
    # from model import *
    '''
    import ****
    '''
    wrapper = Train_wrapper(config_class, config_num)
    wrapper.start()

    print("hello deep learning !")