import torch
from torch import nn

from models.perception_loss import vgg16_feat, perceptual_loss
from .base_model import BaseModel
from . import networks
from parse_config import ConfigParser
import parse_config


class Pix2pixModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        # parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_digesting_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--lambda_digesting_perceptual', type=float, default=10.0, help='weight for perceptual loss')

            parser.add_argument('--prev_model_path', type=str, default='', help='the path of trained model using previous tasks')
            parser.add_argument('--prev_model_epoch', type=int, default=400, help='the epoch of trained model using previous tasks')

            parser.add_argument('--lambda_G', type=float, default=0.1, help='weight for dadgan G ')
            parser.add_argument('--lambda_D', type=float, default=0.05, help='weight for dadgan D')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.loss_names = ['G_GAN', 'G_L1', 'G_perceptual', 'D_real', 'D_fake']
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain and self.opt.prev_model_path:
            self.load_prev_model()

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            self.vgg_model = vgg16_feat().cuda()
            self.criterion_perceptual = perceptual_loss()

    def load_prev_model(self):
        """Load trained Generator using previous tasks
        """
        nets = [self.netG]
        model_paths = ['{:s}/{:d}_net_G.pth'.format(self.opt.prev_model_path, self.opt.prev_model_epoch)]

        for i in range(len(nets)):
            net = nets[i]
            path = model_paths[i]
            if isinstance(net, torch.nn.DataParallel):
                net = net.module
            print('loading the model from %s' % path)
            state_dict = torch.load(path, map_location=self.device)
            if hasattr(state_dict, '_metadata'):
                del state_dict._metadata
            net.load_state_dict(state_dict)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        if self.opt.dataset_mode.split('_')[-1] != 'split' or (not self.opt.isTrain):
            self.real_A = input['A'].to(self.device)
            self.real_B = input['B'].to(self.device)
            self.image_paths = input['A_paths']
        else:
            self.real_A = input['A_' + str(self.opt.task_num-1)].to(self.device)
            self.real_B = input['B_' + str(self.opt.task_num-1)].to(self.device)
            self.image_paths = input['A_paths_' + str(self.opt.task_num-1)]

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # for current task

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * self.opt.lambda_D
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B)
        pred_feat = self.vgg_model(self.fake_B)
        target_feat = self.vgg_model(self.real_B)
        self.loss_G_perceptual = self.criterion_perceptual(pred_feat, target_feat)

        self.loss_G = (self.loss_G_GAN + self.loss_G_L1 * self.opt.lambda_digesting_L1 + self.loss_G_perceptual * self.opt.lambda_digesting_perceptual)*self.opt.lambda_G
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights