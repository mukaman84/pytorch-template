import argparse
import os
import numpy as np
from tqdm import tqdm
import torch
from utils.parallel import DataParallelModel, DataParallelCriterion
from apex import amp
from apex.parallel import DistributedDataParallel
from dataloaders import make_data_loader
from utils.loss import SegmentationLosses, SegmentationCELosses, SegmentationfocalLosses,FocalLoss,disc_loss
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator
from modeling.SCNN import SCNN
from scipy import misc
import ssl
ssl._create_default_https_context = ssl._create_unverified_context



class Trainer(object):
    def __init__(self, args):
        self.args = args

        # Define Saver
        if args.distributed:
            if args.local_rank ==0:
                self.saver = Saver(args)
                self.saver.save_experiment_config()
                # Define Tensorboard Summary
                self.summary = TensorboardSummary(self.saver.experiment_dir)
                self.writer = self.summary.create_summary()
        else:
            self.saver = Saver(args)
            self.saver.save_experiment_config()
            # Define Tensorboard Summary
            self.summary = TensorboardSummary(self.saver.experiment_dir)
            self.writer = self.summary.create_summary()

        # PATH = args.path
        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)
        # self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)

        # Define network
        model = SCNN(nclass=self.nclass,backbone=args.backbone,output_stride=args.out_stride,cuda = args.cuda,extension=args.ext)


        # Define Optimizer
        # optimizer = torch.optim.SGD(model.parameters(),args.lr, momentum=args.momentum,
        #                             weight_decay=args.weight_decay, nesterov=args.nesterov)
        optimizer = torch.optim.Adam(model.parameters(), args.lr,weight_decay=args.weight_decay)

        # model, optimizer = amp.initialize(model,optimizer,opt_level="O1")

        # Define Criterion
        weight = None
        # criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)
        # self.criterion = SegmentationCELosses(weight=weight, cuda=args.cuda)
        # self.criterion = SegmentationCELosses(weight=weight, cuda=args.cuda)
        # self.criterion = FocalLoss(gamma=0, alpha=[0.2, 0.98], img_size=512*512)
        self.criterion1 = FocalLoss(gamma=5, alpha=[0.2, 0.98], img_size=512 * 512)
        self.criterion2 = disc_loss(delta_v=0.5, delta_d=3.0, param_var=1.0, param_dist=1.0,
                                    param_reg=0.001, EMBEDDING_FEATS_DIMS=21,image_shape=[512,512])

        self.model, self.optimizer = model, optimizer
        
        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)
        # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                            args.epochs, len(self.val_loader),local_rank=args.local_rank)

        # Using cuda
        if args.cuda:
            self.model = self.model.cuda()
            if args.distributed:
                self.model = DistributedDataParallel(self.model)
            # self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            # patch_replication_callback(self.model)


        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            filename = 'checkpoint.pth.tar'
            args.resume = os.path.join(self.saver.experiment_dir, filename)
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            # if args.cuda:
            #     self.model.module.load_state_dict(checkpoint['state_dict'])
            # else:
            self.model.load_state_dict(checkpoint['state_dict'])
            # if not args.ft:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))


    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        max_instances = 1
        for i, sample in enumerate(tbar):
            # image, target = sample['image'], sample['label']
            image, target, ins_target = sample['image'], sample['bin_label'], sample['label']
            # _target = target.cpu().numpy()
            # if np.max(_target) > max_instances:
            #     max_instances = np.max(_target)
            #     print(max_instances)

            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            output = self.model(image)

            # if i % 10==0:
            #     misc.imsave('/mfc/user/1623600/.temp6/train_{:s}_epoch:{}_i:{}.png'.format(str(self.args.distributed),epoch,i),np.transpose(image[0].cpu().numpy(),(1,2,0)))
            #     os.chmod('/mfc/user/1623600/.temp6/train_{:s}_epoch:{}_i:{}.png'.format(str(self.args.distributed),epoch,i),0o777)


            # self.criterion = DataParallelCriterion(self.criterion)
            loss1 = self.criterion1(output, target)
            loss2 = self.criterion2(output, ins_target)

            reg_lambda = 0.01


            loss = loss1 + 10*loss2
            # loss = loss1
            output=output[1]
            # loss.back
            # with amp.scale_loss(loss, self.optimizer) as scaled_loss:
            #     scaled_loss.backward()

            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))

            if self.args.distributed:
                if self.args.local_rank == 0:
                    self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)
            else:
                self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)


            # Show 10 * 3 inference results each epoch
            if i % (num_img_tr / 10) == 0:
                global_step = i + num_img_tr * epoch
                if self.args.distributed:
                    if self.args.local_rank == 0:
                        self.summary.visualize_image(self.writer, self.args.dataset, image, target, output, global_step)
                else:
                    self.summary.visualize_image(self.writer, self.args.dataset, image, target, output, global_step)

        if self.args.distributed:
            if self.args.local_rank == 0:
                self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        else:
            self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)

        if self.args.local_rank == 0:
            print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
            print('Loss: %.3f' % train_loss)

        # if self.args.distributed:
        #     if self.args.local_rank == 0:
        #         if self.args.no_val:
        #             # save checkpoint every epoch
        #             is_best = False
        #             self.saver.save_checkpoint({
        #                 'epoch': epoch + 1,
        #                 'state_dict': self.model.module.state_dict(),
        #                 'optimizer': self.optimizer.state_dict(),
        #                 'best_pred': self.best_pred,
        #             }, is_best)
        #     else:
        #         if self.args.no_val:
        #             # save checkpoint every epoch
        #             is_best = False
        #             self.saver.save_checkpoint({
        #                 'epoch': epoch + 1,
        #                 'state_dict': self.model.module.state_dict(),
        #                 'optimizer': self.optimizer.state_dict(),
        #                 'best_pred': self.best_pred,
        #             }, is_best)



    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            # image, target = sample['image'], sample['label']
            image, target = sample['image'], sample['bin_label']
            a= target.numpy()
            aa_max = np.max(a)
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
            loss = self.criterion1(output, target)
            test_loss += loss.item()
            instance_seg = output[0].data.cpu().numpy()
            instance_seg = np.squeeze(instance_seg[0])
            instance_seg = np.transpose(instance_seg, (1, 2, 0))
            output = output[1]
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)
            if i % 30==0:
                misc.imsave('/mfc/user/1623600/.temp6/{:s}_val_epoch:{}_i:{}.png'
                            .format(str(self.args.distributed),epoch,i),
                            np.transpose(image[0].cpu().numpy(),(1,2,0))+3*np.asarray(np.stack((pred[0],pred[0],pred[0]),axis=-1),dtype=np.uint8))
                os.chmod('/mfc/user/1623600/.temp6/{:s}_val_epoch:{}_i:{}.png'.format(str(self.args.distributed),epoch,i),0o777)
                temp_instance_seg = np.zeros_like(np.transpose(image[0].cpu().numpy(),(1,2,0)))
                for j in range(21):
                    if j<7:
                        temp_instance_seg[:, :, 0] += instance_seg[:, :, j]
                    elif j<14:
                        temp_instance_seg[:, :, 1] += instance_seg[:, :, j]
                    else:
                        temp_instance_seg[:, :, 2] += instance_seg[:, :, j]

                for k in range(3):
                    temp_instance_seg[:, :, k] = self.minmax_scale(temp_instance_seg[:, :, k])

                instance_seg = np.array(temp_instance_seg, np.uint8)


                misc.imsave('/mfc/user/1623600/.temp6/emb_{:s}_val_epoch:{}_i:{}.png'
                            .format(str(self.args.distributed), epoch, i),instance_seg[...,:3])
                os.chmod(
                    '/mfc/user/1623600/.temp6/emb_{:s}_val_epoch:{}_i:{}.png'.format(str(self.args.distributed), epoch, i),
                    0o777)



        if self.args.distributed:
            if self.args.local_rank == 0:
                # Fast test during the training
                Acc = self.evaluator.Pixel_Accuracy()
                Acc_class = self.evaluator.Pixel_Accuracy_Class()
                mIoU = self.evaluator.Mean_Intersection_over_Union()
                F0 = self.evaluator.F0()
                self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
                self.writer.add_scalar('val/mIoU', mIoU, epoch)
                self.writer.add_scalar('val/Acc', Acc, epoch)
                self.writer.add_scalar('val/Acc_class', Acc_class, epoch)

                if self.args.local_rank == 0:
                    print('Validation:')
                    print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
                    print("Acc:{}, Acc_class:{}, mIoU:{}".format(Acc, Acc_class, mIoU))
                    print('Loss: %.3f' % test_loss)

                new_pred = F0
                if new_pred > self.best_pred:
                    is_best = True
                    self.best_pred = new_pred
                    self.saver.save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'best_pred': self.best_pred,
                    }, is_best)
        else:
            # Fast test during the training
            Acc = self.evaluator.Pixel_Accuracy()
            Acc_class = self.evaluator.Pixel_Accuracy_Class()
            mIoU = self.evaluator.Mean_Intersection_over_Union()
            F0 = self.evaluator.F0()
            self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
            self.writer.add_scalar('val/mIoU', mIoU, epoch)
            self.writer.add_scalar('val/Acc', Acc, epoch)
            self.writer.add_scalar('val/Acc_class', Acc_class, epoch)

            if self.args.local_rank == 0:
                print('Validation:')
                print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
                print("Acc:{}, Acc_class:{}, mIoU:{}".format(Acc, Acc_class, mIoU))
                print('Loss: %.3f' % test_loss)

            new_pred = F0
            if new_pred > self.best_pred:
                is_best = True
                self.best_pred = new_pred
                self.saver.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'best_pred': self.best_pred,
                }, is_best)

    def minmax_scale(self,input_arr):
        """

        :param input_arr:
        :return:
        """
        min_val = np.min(input_arr)
        max_val = np.max(input_arr)

        output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)

        return output_arr

def main():
    parser = argparse.ArgumentParser(description="PyTorch SCNN Training")
    parser.add_argument('--distributed', type=bool, default=True,
                        help='backbone name (default: resnet)')
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    # parser.add_argument('--path', type=str, default='/mfc/data/compressed/Cityscapes/download',
    #                     help='path of cityscapes')

    parser.add_argument('--path', type=str, default='/mfc/data/mobis/real/91_poc_ms',
                                            help='path of LaneMobis')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='LaneMobis',
                        help='dataset name (default: cityscapes)')
    parser.add_argument('--workers', type=int, default=8,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=512,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=512,
                        help='crop image size')
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal'],
                        help='loss func type (default: ce)')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=6,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=1,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.8,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        metavar='M', help='w-decay (default: 4e-4)')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='whether use nesterov (default: False)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0,1)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')
    parser.add_argument('--ext', default='unet',
                        help='extension branch name (default: unet)')



    args = parser.parse_args()
    if args.distributed != True:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    else:
        pass

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        epoches = {
            'cityscapes': 200,
        }
        args.epochs = epoches[args.dataset.lower()]

    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    if args.lr is None:
        lrs = {
            'cityscapes': 0.01,
        }
        args.lr = lrs[args.dataset.lower()] / (4 * len(args.gpu_ids)) * args.batch_size

    if args.distributed:
        # FOR DISTRIBUTED:  Set the device according to local_rank.
        torch.cuda.set_device(args.local_rank)

        # FOR DISTRIBUTED:  Initialize the backend.  torch.distributed.launch will provide
        # environment variables, and requires that you use init_method=`env://`.
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
    else:
        args.batch_size = 2
        args.test_batch_size = 2


    if args.checkname is None:
        args.checkname = 'SCNN-'+str(args.backbone)
    print(args)

    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    if args.distributed and args.local_rank == 0:
        print('Starting Epoch:', trainer.args.start_epoch)
        print('Total Epoches:', trainer.args.epochs)
    elif not args.distributed :
        print('Starting Epoch:', trainer.args.start_epoch)
        print('Total Epoches:', trainer.args.epochs)

    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.validation(epoch)

    trainer.writer.close()

if __name__ == "__main__":
   main()
