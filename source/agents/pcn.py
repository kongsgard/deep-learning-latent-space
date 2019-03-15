from tensorboardX import SummaryWriter
import torch
from tqdm import tqdm
import visdom

from agents.base import BaseAgent
from graphs.losses.chamfer_loss import ChamferDistance
from graphs.models.pcn import PCN
from datasets.shapenet_point_cloud import ShapeNetPointCloudDataLoader
from utils.metrics import AverageMeter
from utils.misc import print_cuda_statistics
from utils.pcd.pcd_utils import plot_completion_results


class PointCompletionNetworkAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)

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

        # Define model
        self.model = PCN(self.config)
        self.model = self.model.to(self.device)

        # Define dataloader
        self.train_dataloader = ShapeNetPointCloudDataLoader(self.config, 
                                                             dataset_mode='train')
        self.validate_dataloader = ShapeNetPointCloudDataLoader(self.config,
                                                                dataset_mode='valid')

        # Define optimizer and scheduler
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.config.learning_rate,
                                          weight_decay=self.config.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, 0.97)

        # Define criterion
        self.criterion = ChamferDistance()
        self.criterion = self.criterion.to(self.device)

        # Initialize counter
        self.current_epoch = 0
        self.current_iteration = 0

        # Load model from the latest checkpoint.
        # If none can be found, start from scratch.
        self.load_checkpoint(self.config.checkpoint_file)

        # Visualization in visdom during training
        self.vis = visdom.Visdom()

        # Summary Writer
        self.summary_writer = SummaryWriter(log_dir=self.config.summary_dir,
                                            comment='PCN')

    def update_loss(self, coarse, fine, gt_points):
        # TODO: Add summaries

        #gt_downsampled = gt_points[:, :coarse.shape[1], :] # Possibly not necessary if only Chamfer Distance is used

        if self.current_epoch >= 70:
            self.config.alpha = 0.1
        elif self.current_epoch >= 140:
            self.config.alpha = 0.5
        elif self.current_epoch >= 350:
            self.config.alpha = 1.0

        dist1, dist2 = self.criterion(coarse, gt_points)
        loss_coarse = (torch.mean(dist1)) + (torch.mean(dist2))

        dist1, dist2 = self.criterion(fine, gt_points)
        loss_fine = (torch.mean(dist1)) + (torch.mean(dist2))

        loss = loss_coarse + self.config.alpha * loss_fine

        return loss, loss_coarse, loss_fine

    def load_checkpoint(self, file_name):
        filename = self.config.checkpoint_dir + file_name
        try:
            self.logger.info("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)

            self.current_epoch = checkpoint['epoch']
            self.current_iteration = checkpoint['iteration']
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            self.logger.info("Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n"
                             .format(self.config.checkpoint_dir, checkpoint['epoch'], checkpoint['iteration']))
        except OSError as e:
            self.logger.info("No checkpoint exists from '{}'. Skipping...".format(self.config.checkpoint_dir))
            self.logger.info("**First time to train**")

    def save_checkpoint(self, file_name="checkpoint.pth.tar", is_best=0):
        state = {
            'epoch': self.current_epoch,
            'iteration': self.current_iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
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
            self.scheduler.step()
            self.train_one_epoch()
            self.validate()
            self.save_checkpoint()

    def train_one_epoch(self):
        # Initialize tqdm batch
        tqdm_batch = tqdm(self.train_dataloader.loader,
                          total=self.train_dataloader.num_iterations,
                          desc="Epoch -{}-".format(self.current_epoch))

        model_loss_epoch = AverageMeter()
        loss_coarse_epoch = AverageMeter()
        loss_fine_epoch = AverageMeter()

        for curr_it, x in enumerate(tqdm_batch):
            ids, input_points, gt_points = x

            self.optimizer.zero_grad()
            
            if self.cuda:
                input_points = input_points.cuda(non_blocking=self.config.async_loading)
                gt_points = gt_points.cuda(non_blocking=self.config.async_loading)

            if input_points.size()[0] != int(self.config.batch_size):
                # print("Batch_size != {} - dropping last incompatible batch".format(int(self.config.batch_size)))
                continue

            coarse, fine = self.model(input_points)

            loss, loss_coarse, loss_fine = self.update_loss(coarse, fine, gt_points)
            loss.backward()
            self.optimizer.step()

            # Update and log the current loss
            model_loss_epoch.update(loss.item())
            #loss_coarse_epoch.update(loss_coarse.item())
            #loss_fine_epoch.update(loss_fine.item())

            self.current_iteration += 1

            self.summary_writer.add_scalar("iteration/loss", loss.item(), self.current_iteration)

            # Visualize
            if self.config.visualize and curr_it % 200 == 0:
                plot_completion_results(self.vis,
                                        input_points.contiguous()[0].data.cpu(),
                                        coarse.contiguous()[0].data.cpu(),
                                        fine.contiguous()[0].data.cpu(),
                                        gt_points[0].data.cpu()
                                        )

        tqdm_batch.close()

        self.logger.info("Training at epoch-{:d} | Network loss: {:.3f}"
                         .format(self.current_epoch, model_loss_epoch.val))
        self.summary_writer.add_scalar("epoch-training/loss", model_loss_epoch.val, self.current_epoch)

    def validate(self):
        self.model.eval()

        tqdm_batch = tqdm(self.validate_dataloader.loader,
                          total=self.validate_dataloader.num_iterations,
                          desc="Validation at -{}-".format(self.current_epoch))

        model_loss_epoch = AverageMeter()

        with torch.no_grad():
            for curr_it, x in enumerate(tqdm_batch):
                ids, input_points, gt_points = x
                
                if self.cuda:
                    input_points = input_points.cuda(non_blocking=self.config.async_loading)
                    gt_points = gt_points.cuda(non_blocking=self.config.async_loading)

                coarse, fine = self.model(input_points)
                loss, loss_coarse, loss_fine = self.update_loss(coarse, fine, gt_points)

                model_loss_epoch.update(loss.item())
        
        self.logger.info("Validation at epoch-{:d} | Network loss: {:.3f}"
                         .format(self.current_epoch, model_loss_epoch.val))
        self.summary_writer.add_scalar("epoch-validation/loss", model_loss_epoch.val, self.current_epoch)
        
        tqdm_batch.close()
            

    def finalize(self):
        self.logger.info("Agent has finished running - wait to finalize...")
        self.save_checkpoint()

        self.summary_writer.export_scalars_to_json("{}all_scalars.json".format(self.config.summary_dir))
        self.summary_writer.close()

        self.train_dataloader.finalize()