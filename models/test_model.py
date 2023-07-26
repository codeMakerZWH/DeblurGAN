import torch

from torch.autograd import Variable
from collections import OrderedDict
import util.util as util
from .base_model import BaseModel
from . import networks
from PIL import Image
import numpy as np

class TestModel(BaseModel):
    def name(self):
        return 'TestModel'

    def __init__(self, opt):
        assert(not opt.isTrain)
        super(TestModel, self).__init__(opt)
        self.input_A = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)

        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt.norm, not opt.no_dropout, self.gpu_ids, False,
                                      opt.learn_residual)
        which_epoch = opt.which_epoch
        self.load_network(self.netG, 'G', which_epoch)

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        print('-----------------------------------------------')

    def set_input(self, input):
        # we need to use single_dataset mode
        input_A = input['A']
        input_B= input['B']
        temp = self.input_A.clone()
        temp.resize_(input_A.size()).copy_(input_A)
        self.input_B = input_B
        self.input_A = temp
        self.image_paths = input['A_paths']

    def test(self):
        with torch.no_grad():
            self.real_A = Variable(self.input_A)
            self.real_B = Variable(self.input_B)
            self.fake_B = self.netG.forward(self.real_A)

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        gt = util.tensor2im(self.input_B.data)
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('GT', gt)])

    def show_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        # 创建PIL图像对象
        real_A_img = Image.fromarray(np.uint8(real_A))
        fake_B_img = Image.fromarray(np.uint8(fake_B))

        # 显示图像
        real_A_img.show(title='Real A')
        fake_B_img.show(title='Fake B')