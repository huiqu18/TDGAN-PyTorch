import torch
from torch import nn

from models.perception_loss import vgg16_feat, perceptual_loss
from .base_model import BaseModel
from . import networks
from parse_config import ConfigParser
import parse_config


class TDGANModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
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

        For pix2pix, we do not use image buffer
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        # parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_digesting_L1', type=float, default=100.0, help='weight for digesting L1 loss')
            parser.add_argument('--lambda_digesting_perceptual', type=float, default=1.0, help='weight for digesting perceptual loss')

            parser.add_argument('--prev_model_path', type=str, default='', help='the path of trained model using previous tasks')
            parser.add_argument('--prev_model_epoch', type=int, default=400, help='the epoch of trained model using previous tasks')
            parser.add_argument('--lambda_reminding_L1', type=float, default=10.0, help='weight for reminding L1 loss')
            parser.add_argument('--lambda_reminding_perceptual', type=float, default=1.0, help='weight for reminding perceptual loss')

            parser.add_argument('--lambda_G', type=float, default=0.1, help='weight for dadgan G ')
            parser.add_argument('--lambda_D', type=float, default=0.05, help='weight for dadgan D')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        if self.opt.task_num == 1:
            self.loss_names = ['G_GAN', 'G_L1', 'G_perceptual', 'D_real', 'D_fake']
            self.visual_names = ['real_A_cur', 'fake_B_cur', 'real_B_cur']
        else:
            self.loss_names = ['G_GAN', 'G_L1', 'G_perceptual', 'D_real', 'D_fake', 'reminding_L1_all', 'reminding_perceptual_all']
            self.visual_names = ['real_A_prev', 'real_B_prev', 'fake_B_cur_prev', 'fake_B_prev', 'real_A_cur', 'fake_B_cur', 'real_B_cur']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        self.netG_prev = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                           not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images
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
        nets = [self.netG_prev, self.netG]
        model_paths = ['{:s}/{:d}_net_G.pth'.format(self.opt.prev_model_path, self.opt.prev_model_epoch),
                       '{:s}/{:d}_net_G.pth'.format(self.opt.prev_model_path, self.opt.prev_model_epoch)]
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
        self.real_A = []
        self.real_B = []
        self.image_paths = []
        for i in range(self.opt.task_num):
            self.real_A.append(input['A_' + str(i)].to(self.device))
            self.real_B.append(input['B_' + str(i)].to(self.device))
            self.image_paths.append(input['A_paths_' + str(i)])

        self.real_A_cur = self.real_A[self.opt.task_num-1]  # real label for current task
        self.real_B_cur = self.real_B[self.opt.task_num-1]  # real image for current task
        if self.opt.task_num > 1:
            self.real_A_prev = self.real_A[0]  # real label for the first task
            self.real_B_prev = self.real_B[0]  # real image for the first task

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A[self.opt.task_num - 1])  # for current task

        # for previous tasks
        self.fake_B_curG = []
        self.fake_B_prevG = []
        for i in range(self.opt.task_num-1):  # fake images from labels of previous tasks using both previous G and current G
            self.fake_B_curG.append(self.netG(self.real_A[i]))
            self.fake_B_prevG.append(self.netG_prev(self.real_A[i]))

        self.fake_B_cur = self.fake_B
        if self.opt.task_num > 1:
            self.fake_B_cur_prev = self.fake_B_curG[0]  # fake image from label of the first task using current G
            self.fake_B_prev = self.fake_B_prevG[0]  # fake image from label of the first task using previous G

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A[self.opt.task_num-1], self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A[self.opt.task_num-1], self.real_B[self.opt.task_num-1]), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * self.opt.lambda_D
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # ----- previous Ds are not available ----- #
        # for current task
        fake_AB = torch.cat((self.real_A[self.opt.task_num-1], self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B[self.opt.task_num-1])
        pred_feat = self.vgg_model(self.fake_B)
        target_feat = self.vgg_model(self.real_B[self.opt.task_num-1])
        self.loss_G_perceptual = self.criterion_perceptual(pred_feat, target_feat)

        # for previous tasks
        self.loss_reminding_L1 = []
        self.loss_reminding_perceptual = []
        for i in range(self.opt.task_num-1):
            self.loss_reminding_L1.append(self.criterionL1(self.fake_B_curG[i], self.fake_B_prevG[i]))
            pred_feat_G = self.vgg_model(self.fake_B_curG[i])
            pred_feat_G_prev = self.vgg_model(self.fake_B_prevG[i])
            self.loss_reminding_perceptual.append(self.criterion_perceptual(pred_feat_G, pred_feat_G_prev))

        self.loss_reminding_L1_all = None
        self.loss_reminding_perceptual_all = None
        for i in range(len(self.loss_reminding_L1)):
            if self.loss_reminding_L1_all is None:
                self.loss_reminding_L1_all = self.loss_reminding_L1[i]
                self.loss_reminding_perceptual_all = self.loss_reminding_perceptual[i]
            else:
                self.loss_reminding_L1_all += self.loss_reminding_L1[i]
                self.loss_reminding_perceptual_all += self.loss_reminding_perceptual[i]

        if self.opt.task_num == 1:
            self.loss_G = (self.loss_G_GAN + self.loss_G_L1 * self.opt.lambda_digesting_L1 + self.loss_G_perceptual * self.opt.lambda_digesting_perceptual)*self.opt.lambda_G
        else:  # digesting loss = loss_G_GAN + loss_G_L1 + loss_G_perceptual,  reminding loss = loss_reminding_L1 + loss_reminding_perceptual
            self.loss_G = (self.loss_G_GAN + self.loss_G_L1 * self.opt.lambda_digesting_L1 + self.loss_G_perceptual * self.opt.lambda_digesting_perceptual
                           + self.loss_reminding_L1_all * self.opt.lambda_reminding_L1 + self.loss_reminding_perceptual_all * self.opt.lambda_reminding_perceptual)*self.opt.lambda_G
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