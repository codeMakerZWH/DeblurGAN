import time
import os
from multiprocessing import freeze_support

from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from pdb import set_trace as st
from util import html
from util.metrics import PSNR, SSIM

from PIL import Image

from skimage.metrics import mean_squared_error as compare_mse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim


opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

opt.dataroot = r'E:\Python porject\DeblurGAN-master\dataset\carton_AB\valid'
# opt.dataroot = r'E:\DataSet\OT_cut\Carton\test\blur'
opt.model = 'test'
opt.dataset_mode = 'single'
opt.learn_residual = True
opt.which_model_netG = "unet_256"
opt.resize_or_crop = "none"




data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
# test
def test():
	avgPSNR = 0.0
	avgSSIM = 0.0
	counter = 0

	for i, data in enumerate(dataset):
		if i >= opt.how_many:
			break
		counter = i
		model.set_input(data)
		model.test()
		# model.show_current_visuals()
		visuals = model.get_current_visuals()




		avgPSNR += compare_psnr(visuals['fake_B'], visuals['GT'])
		avgSSIM += compare_ssim(visuals['fake_B'], visuals['GT'], multichannel=True)
		img_path = model.get_image_paths()
		print('process image... %s' % img_path)
		visualizer.save_images(webpage, visuals, img_path)

	avgPSNR /= counter
	avgSSIM /= counter
	print('PSNR = %f, SSIM = %f' %
					  (avgPSNR, avgSSIM))

	webpage.save()

if __name__ == '__main__':
	freeze_support()
	test()