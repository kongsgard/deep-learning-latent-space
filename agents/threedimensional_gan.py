import shutil
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

        # Define optimizers
        self.g_solver = torch.optim.Adam(self.g_net.parameters(),
                                         lr=self.config.g_learning_rate, betas=(self.config.beta1, self.config.beta2))
        self.d_solver = torch.optim.Adam(self.d_net.parameters(),
                                         lr=self.config.d_learning_rate, betas=(self.config.beta1, self.config.beta2))

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

        # Visdom 3D Plotting
        self.vis = visdom.Visdom()

        # Summary Writer
        self.generator_summary_writer = SummaryWriter(log_dir=self.config.summary_dir + '/generator',
                                                      comment='3DGAN')
        self.discriminator_summary_writer = SummaryWriter(log_dir=self.config.summary_dir + '/discriminator',
                                                          comment='3DGAN')

    def load_checkpoint(self, file_name):
        pass

    def save_checkpoint(self, file_name="checkpoint.pth.tar", is_best=0):
        state = {
            'epoch': self.current_epoch,
            'iteration': self.current_iteration,
            'G_state_dict': self.g_net.state_dict(),
            'G_optimizer': self.g_solver.state_dict(),
            'D_state_dict': self.d_net.state_dict(),
            'D_optimizer': self.d_solver.state_dict(),
        }
        # Save the state
        torch.save(state, self.config.checkpoint_dir + file_name)
        # If it is the best copy it to another file 'model_best.pth.tar'
        if is_best:
            shutil.copyfile(self.config.checkpoint_dir + file_name,
                            self.config.checkpoint_dir + 'model_best.pth.tar')

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

    def train_one_epoch(self):
        # Initialize tqdm batch
        tqdm_batch = tqdm(self.dataloader.loader, total=self.dataloader.num_iterations,
                          desc="epoch-{}-".format(self.current_epoch))

        g_loss_epoch = AverageMeter()
        d_loss_epoch = AverageMeter()

        for curr_it, x in enumerate(tqdm_batch):
            z = generate_fake_noise(self.config)
            real_labels = torch.ones(self.config.batch_size).view(-1, 1, 1, 1, 1)
            fake_labels = torch.zeros(self.config.batch_size).view(-1, 1, 1, 1, 1)

            if self.cuda:
                x = x.cuda(async=self.config.async_loading)
                z = z.cuda(async=self.config.async_loading)
                real_labels = real_labels.cuda(async=self.config.async_loading)
                fake_labels = fake_labels.cuda(async=self.config.async_loading)

            if x.size()[0] != int(self.config.batch_size):
                # print("Batch_size != {} - dropping last incompatible batch".format(int(self.config.batch_size)))
                continue

            # === Train the discriminator ===#
            # Train with real data
            d_real_out = self.d_net(x)
            d_real_loss = self.loss(d_real_out, real_labels)

            # Train with fake data
            g_fake_out = self.g_net(z)
            d_fake_out = self.d_net(g_fake_out.detach())
            d_fake_loss = self.loss(d_fake_out, fake_labels)

            d_loss = d_real_loss + d_fake_loss

            d_real_acu = torch.ge(d_real_out.squeeze(), 0.5).float()
            d_fake_acu = torch.le(d_fake_out.squeeze(), 0.5).float()
            d_total_acu = torch.mean(torch.cat((d_real_acu, d_fake_acu), 0))

            if d_total_acu <= self.config.d_threshold:
                self.d_net.zero_grad()
                d_loss.backward()
                self.d_solver.step()

            # === Train the generator ===#
            z = generate_fake_noise(self.config)
            if self.cuda:
                z = z.cuda(async=self.config.async_loading)

            g_fake_out = self.g_net(z)
            d_fake_out = self.d_net(g_fake_out)
            g_loss = self.loss(d_fake_out, real_labels)

            self.d_net.zero_grad()
            self.g_net.zero_grad()
            g_loss.backward()
            self.g_solver.step()

            g_loss_epoch.update(g_loss.item())
            d_loss_epoch.update(d_loss.item())

            self.current_iteration += 1

            # Plot shapes in visdom
            if curr_it == 0:  # Todo: Refactor
                plot_voxels_in_visdom(torch.Tensor.numpy(x.cpu()[0]), self.vis, "shape", "true")
                plot_voxels_in_visdom(torch.Tensor.numpy(g_fake_out.detach().cpu()[0][0]), self.vis, "shape", "fake")

            self.generator_summary_writer.add_scalar("iteration/loss", g_loss.item(), self.current_iteration)
            self.discriminator_summary_writer.add_scalar("iteration/loss", d_loss.item(), self.current_iteration)
            self.discriminator_summary_writer.add_scalar("iteration/real_loss", d_real_loss.item(), self.current_iteration)
            self.discriminator_summary_writer.add_scalar("iteration/fake_loss", d_fake_loss.item(), self.current_iteration)

        tqdm_batch.close()

        self.logger.info("Training at epoch-{:d} | Generator loss: {:.3f} - Discriminator loss: {:.3f}"
                         .format(self.current_epoch, g_loss_epoch.val, d_loss_epoch.val))
        self.generator_summary_writer.add_scalar("epoch/loss", g_loss_epoch.val, self.current_epoch)
        self.discriminator_summary_writer.add_scalar("epoch/loss", d_loss_epoch.val, self.current_epoch)

    def validate(self):
        pass

    def finalize(self):
        self.logger.info("Agent has finished running - wait to finalize...")
        self.save_checkpoint()

        self.generator_summary_writer.export_scalars_to_json("{}all_scalars.json".format(self.config.summary_dir))
        self.generator_summary_writer.close()
        self.discriminator_summary_writer.export_scalars_to_json("{}all_scalars.json".format(self.config.summary_dir))
        self.discriminator_summary_writer.close()

        self.dataloader.finalize()
