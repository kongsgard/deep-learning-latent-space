from tqdm import tqdm

from tensorboardX import SummaryWriter
import torch
import visdom

from agents.base import BaseAgent
from graphs.losses.bce import BinaryCrossEntropy
from graphs.models.threed_gan_generator import Generator
from graphs.models.threed_gan_discriminator import Discriminator
from datasets.shapenet import ShapeNetDataLoader
from utils.metrics import AverageMeter
from utils.misc import print_cuda_statistics
from utils.voxel_utils import generate_fake_noise, plot_voxels_in_visdom


class ThreeDimensionalGANAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)

        # Define models (generator and discriminator)
        self.g_net = Generator(self.config)
        self.d_net = Discriminator(self.config)

        # Define dataloader
        self.dataloader = ShapeNetDataLoader(self.config)

        # Define loss
        self.loss = BinaryCrossEntropy()

        # Initialize counter
        self.current_epoch = 0
        self.current_iteration = 0

        # Set cuda flag
        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda and not self.config.cuda:
            self.logger.info("WARNING: You have a CUDA device - enable it!")
        self.cuda = self.is_cuda & self.config.cuda

        # Set the manual seed for torch
        if self.cuda:
            self.device = torch.device("cuda")
            torch.cuda.set_device(self.config.gpu_device)
            torch.cuda.manual_seed_all(self.config.seed)
            self.logger.info("Program will run on ***GPU-CUDA***")
            print_cuda_statistics()
        else:
            self.device = torch.device("cpu")
            torch.manual_seed(self.config.seed)
            self.logger.info("Program will run on ***CPU***")

        self.g_net = self.g_net.to(self.device)
        self.d_net = self.d_net.to(self.device)
        self.loss = self.loss.to(self.device)

        # Summary Writer
        self.summary_writer = SummaryWriter(log_dir=self.config.summary_dir, comment='3DGAN')

    def load_checkpoint(self, file_name):
        pass

    def save_checkpoint(self, file_name="checkpoint.pth.tar", is_best=0):
        pass

    def run(self):
        try:
            self.train()

        except KeyboardInterrupt:
            self.logger.info("CTRL+C was pressed - wait to finalize...")

    def train(self):
        for epoch in range(self.current_epoch, self.config.max_epoch):
            self.current_epoch = epoch
            self.train_one_epoch()
            self.save_checkpoint()

            break  # TODO: Remove

    def train_one_epoch(self):
        # Initialize tqdm batch
        tqdm_batch = tqdm(self.dataloader.loader, total=self.dataloader.num_iterations,
                          desc="epoch-{}-".format(self.current_epoch))

        g_epoch_loss = AverageMeter()
        d_epoch_loss = AverageMeter()

        vis = visdom.Visdom()  # TODO: Move
        for curr_it, x in enumerate(tqdm_batch):


            z = generate_fake_noise(self.config)
            real_labels = torch.ones(self.config.batch_size).view(-1, 1, 1, 1, 1)
            fake_labels = torch.zeros(self.config.batch_size).view(-1, 1, 1, 1, 1)

            if self.cuda:
                x = x.cuda(async=self.config.async_loading)
                z = z.cuda(async=self.config.async_loading)
                real_labels = real_labels.cuda(async=self.config.async_loading)
                fake_labels = fake_labels.cuda(async=self.config.async_loading)

            # === Train the discriminator ===
            # Train with real data
            self.d_net.zero_grad()
            d_real_out = self.d_net(x)
            d_real_loss = self.loss(d_real_out, real_labels)

            # Train with fake data
            g_fake_out = self.g_net(z)
            d_fake_out = self.d_net(g_fake_out.detach())
            d_fake_loss = self.loss(d_fake_out, fake_labels)

            if curr_it == 0:  # Todo: Refactor
                plot_voxels_in_visdom(torch.Tensor.numpy(x.cpu()[0]), vis, "shape", "true")
                plot_voxels_in_visdom(torch.Tensor.numpy(g_fake_out.detach().cpu()[0][0]), vis, "shape", "fake")

            break  # TODO: Remove

        g_epoch_loss.update(100)  # TODO: Fix
        d_epoch_loss.update(100)  # TODO: Fix

        tqdm_batch.close()

        self.logger.info("Training at epoch-" + str(self.current_epoch) + " | " + "Generator loss: " + str(
            g_epoch_loss.val) + " - Discriminator loss: " + str(d_epoch_loss.val))

    def validate(self):
        pass

    def finalize(self):
        self.logger.info("Agent has finished running - wait to finalize...")
        self.save_checkpoint()
        self.summary_writer.export_scalars_to_json("{}all_scalars.json".format(self.config.summary_dir))
        self.summary_writer.close()
        self.dataloader.finalize()