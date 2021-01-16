from torch.utils.tensorboard import SummaryWriter
import time,sys,shutil
import torchvision,torch
from dominate.tags import meta, h3, table, tr, td, p, a, img, br,h1
import os,shutil,zipfile,dominate
from subprocess import Popen, PIPE
from base_code import util
import numpy as np

if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError

class Visualizer():

    def __init__(self,opt,display_in_train = False):
        #self.writer = SummaryWriter(os.path.join(default_dir,model.name, model.time))
        self.name = opt.name
        self.opt = opt

        if opt.inference:
            self.counter = 0
            self.html = HTML(opt)
        else:
            # html
            self.use_html = opt.use_html
            # visdom
            self.display_id = opt.display_id
            self.win_size = opt.display_winsize
            self.port = opt.display_port
            if self.display_id > 0:  # connect to a visdom server given <display_port> and <display_server>
                import visdom
                self.ncols = opt.display_ncols
                self.vis = visdom.Visdom(server=opt.display_server, port=opt.display_port, env=opt.display_env)
                if not self.vis.check_connection():
                    self.create_visdom_connections()

            # tensorboard
            if opt.use_tensorboard:
                if os.path.exists(opt.tensorboard_dir):
                    shutil.rmtree(opt.tensorboard_dir,ignore_errors=True)
                self.writer = SummaryWriter(opt.tensorboard_dir)

            if display_in_train:
                p = os.path.join(opt.web_dir,self.name)
                if not os.path.exists(p) :
                    os.makedirs(p)
                self.path_display_in_train = p
                self.imgs_save_path = os.path.join(p,'images')
                self.html = HTML(p, 'display_in_train_html')

            # create a logging file to store training losses
            self.log_name = os.path.join(opt.checkpoints_dir, 'loss_log.txt')

            with open(self.log_name, "a") as log_file:
                now = time.strftime("%c")
                log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        """Reset the self.saved status"""
        self.saved = False

    def create_visdom_connections(self):
        """If the program could not connect to Visdom server, this function will start a new server at port < self.port > """
        cmd = sys.executable + ' -m visdom.server -p %d &>/dev/null &' % self.port
        print('\n\nCould not connect to Visdom server. \n Trying to start a server....')
        print('Command: %s' % cmd)
        Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)

    def plot_loss(self,loss_dict,iteration_or_epoch):
        self.writer.add_scalars(self.name,loss_dict,iteration_or_epoch)

    def display_current_results(self,visual_im):

        if self.display_id > 0:  # show images in the browser using visdom
            ncols = self.ncols
            if ncols > 0:        # show all the images in one visdom panel
                ncols = min(ncols, len(visual_im))
                h, w = next(iter(visual_im.values())).shape[:2]
                table_css = """<style>
                        table {border-collapse: separate; border-spacing: 4px; white-space: nowrap; text-align: center}
                        table td {width: % dpx; height: % dpx; padding: 4px; outline: 4px solid black}
                        </style>""" % (w, h)  # create a table css
                # create a table of images.
                title = self.name
                label_html = ''
                label_html_row = ''
                images = []
                idx = 0
                for label, image in visual_im.items():
                    image_numpy = util.tensor2im(image,size = self.opt.size)
                    label_html_row += '<td>%s</td>' % label
                    images.append(image_numpy.transpose([2, 0, 1]))
                    idx += 1
                    if idx % ncols == 0:
                        label_html += '<tr>%s</tr>' % label_html_row
                        label_html_row = ''
                white_image = np.ones_like(image_numpy.transpose([2, 0, 1])) * 255
                while idx % ncols != 0:
                    images.append(white_image)
                    label_html_row += '<td></td>'
                    idx += 1
                if label_html_row != '':
                    label_html += '<tr>%s</tr>' % label_html_row
                try:
                    self.vis.images(images, nrow=ncols, win=self.display_id + 1,
                                    padding=2, opts=dict(title=title + ' images'))
                    label_html = '<table>%s</table>' % label_html
                    self.vis.text(table_css + label_html, win=self.display_id + 2,
                                  opts=dict(title=title + ' labels'))
                except VisdomExceptionBase:
                    self.create_visdom_connections()

            else:     # show each image in a separate visdom panel;
                idx = 1
                try:
                    for label, image in visual_im.items():
                        image_numpy = util.tensor2im(image)
                        self.vis.image(image_numpy.transpose([2, 0, 1]), opts=dict(title=label),
                                       win=self.display_id + idx)
                        idx += 1
                except VisdomExceptionBase:
                    self.create_visdom_connections()

    def display_results(self, visual_im):
        header = self.opt.header
        path = self.opt.web_dir
        # save images to the disk
        self.counter += 1
        self.html.add_header("\n")
        ims, txts, links = [], [], []
        imgs_save_path = os.path.join(path, 'images', header)
        os.makedirs(imgs_save_path, exist_ok=True)
        self.html.add_header(f"{self.counter}.jpg")
        i = 1
        for label, image in visual_im.items():
            img_name = f'{self.counter}_{i}.jpg'
            # image_numpy = util.tensor2im(image)
            img_path = os.path.join(imgs_save_path, img_name)
            # util.save_image(image_numpy, img_path)
            torchvision.utils.save_image(image, img_path, normalize=True)

            ims.append(os.path.join("images", header, img_name))
            txts.append(label)
            links.append(os.path.join("images", header, img_name))
            i += 1

        self.html.add_images(ims, txts, links)
        self.html.save()

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, iters, t_comp,losses,print_in_stdout = True):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, iters, t_comp)

        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)
        if print_in_stdout:
            print(message)  # print the message

        if not self.opt.debug:
            with open(self.log_name, "a") as log_file:
                log_file.write('%s\n' % message)  # save the message

class HTML:
    """This HTML class allows us to save images and write texts into a single HTML file.

     It consists of functions such as <add_header> (add a text header to the HTML file),
     <add_images> (add a row of images to the HTML file), and <save> (save the HTML to the disk).
     It is based on Python library 'dominate', a Python library for creating and manipulating HTML documents using a DOM API.
    """

    def __init__(self,args,refresh=0):
        """Initialize the HTML classes

        Parameters:
            web_dir (str) -- a directory that stores the webpage. HTML file will be created at <web_dir>/index.html; images will be saved at <web_dir/images/
            title (str)   -- the webpage name
            refresh (int) -- how often the website refresh itself; if 0; no refreshing
        """
        self.args = args
        self.title = args.header
        self.doc = dominate.document(title=self.title)
        self.add_title()
        self.web_dir = args.web_dir
        self.img_dir = os.path.join(self.web_dir, 'images')
        print(os.path.exists(self.web_dir))
        if not os.path.exists(self.web_dir):
            os.makedirs(self.web_dir)

        if refresh > 0:
            with self.doc.head:
                meta(http_equiv="refresh", content=str(refresh))

    def get_image_dir(self):
        """Return the directory that stores images"""
        return self.img_dir

    def add_title(self):
        """Insert a header to the HTML file

        Parameters:
            text (str) -- the header text
        """
        with self.doc:
            h1(self.title+f'_epoch_{self.args.start_iter}')

    def add_header(self, text):
        """Insert a header to the HTML file

        Parameters:
            text (str) -- the header text
        """
        with self.doc:
            h3(text)

    def add_images(self, ims, txts, links, width=400):
        """add images to the HTML file

        Parameters:
            ims (str list)   -- a list of image paths
            txts (str list)  -- a list of image names shown on the website
            links (str list) --  a list of hyperref links; when you click an image, it will redirect you to a new page
        """
        self.t = table(border=1, style="table-layout: fixed;")  # Insert a table
        self.doc.add(self.t)
        with self.t:
            with tr():
                for im, txt, link in zip(ims, txts, links):
                    with td(style="word-wrap: break-word;", halign="center", valign="top"):
                        with p():
                            with a(href=im):
                                img(style="width:%dpx" % width, src=im)
                            br()
                            p(txt)

    def save(self):
        """save the current content to the HMTL file"""
        html_file = f'{self.web_dir}/{self.title}.html'
        f = open(html_file, 'wt')
        f.write(self.doc.render())
        f.close()

def copy_label():
    source = '../datasets/CelebA-HQ-img'
    target = 'images/train/target'

    if not os.path.exists(target):
        os.mkdir(target)

    for name in os.listdir('images/train/blank'):
        n = name.split('_')[-1]
        shutil.copy(os.path.join(source,n),target)

def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram

def zip_display_in_train():
    model_root = './checkpoint/FI_Module'
    for models in os.listdir(model_root):
        path_to_models = os.path.join(model_root,models)
        checks = list(os.listdir(path_to_models))
        if 'display_in_train' not in checks:
            print("There is no display in train in {} !".format(models))
            continue

        path_to_display = os.path.join(path_to_models, 'display_in_train')
        copy_to = os.path.join('./display_in_train',models)
        shutil.copytree(path_to_display, copy_to)

    f = zipfile.ZipFile('display_in_train.zip','w',zipfile.ZIP_DEFLATED)

    startdir = "./display_in_train"
    for dirpath, dirnames, filenames in os.walk(startdir):
        for filename in filenames:
            f.write(os.path.join(dirpath,filename))
    f.close()
    shutil.rmtree('./display_in_train')

def display_results(html,path,header,visual_im,img_name):
    # save images to the disk
    html.add_header("\n")
    ims, txts, links = [], [], []
    imgs_save_path = os.path.join(path,'images',header)
    os.makedirs(imgs_save_path, exist_ok= True)
    for label, image in visual_im.items():
        img_name =  f'{img_name}.jpg'
        #image_numpy = util.tensor2im(image)
        img_path = os.path.join(imgs_save_path,img_name)
        #util.save_image(image_numpy, img_path)
        torchvision.utils.save_image(image,img_path,normalize = True)

        ims.append(os.path.join("images",header,img_name))
        txts.append(label)
        links.append(os.path.join("images",header,img_name))

    html.add_images(ims, txts, links)
    html.save()


if __name__ == '__main__':

    path = "/data1/yujingbo/repos/web_dir/inference/"
    header = "config_b_1"
    visual_im = {}
    for i in range(8):
        visual_im[i] = torch.ones((3,256,256))
    display_current_results(path,header,visual_im)

    #zip_display_in_train()
