import argparse
import os
import numpy as np
from tqdm import tqdm
import torch
# from utils.parallel import DataParallelModel, DataParallelCriterion
from modeling.postprocess import LanePostprocess
from apex import amp
from apex.parallel import DistributedDataParallel
from dataloaders import make_data_loader
from utils.loss import SegmentationLosses, SegmentationCELosses, SegmentationfocalLosses, FocalLoss, disc_loss
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator
from modeling.SCNN import SCNN
from scipy import misc
from collections import OrderedDict
import ssl
import mvpuai
from mvpuai.annotation.frame import MFrame
from mvpuai.resource.string import MString
import glog as log
from geomdl import BSpline, utilities

from BsplineModel.inference_bs import inference
from BsplineModel.GetBspline import GetBspline_from_sampled_points

ssl._create_default_https_context = ssl._create_unverified_context
class Point(object):

    def __init__(self, x: int, y: int, color_=None, editable: bool=None):
        self.coord = np.array([x, y])
        # self._color = color.POINT if color_ is None else color_
        self.editable = True if editable is None else editable

    # @property
    # def color(self):
    #     return self._color
    #
    # @color.setter
    # def color(self, color_):
    #     self._color = color_
    #
    @property
    def x(self):
        return int(self.coord[0])
    #
    @x.setter
    def x(self, x: int):
        self.coord[0] = x
    #
    @property
    def y(self):
        return int(self.coord[1])
    #
    @y.setter
    def y(self, y: int):
        self.coord[1] = y

class Trainer(object):
    def __init__(self, args):
        self.args = args

        # Define Saver
        if args.distributed:
            if args.local_rank == 0:
                self.saver = Saver(args)
        else:
            self.saver = Saver(args)
            self.saver.save_experiment_config()
            # Define Tensorboard Summary
            self.summary = TensorboardSummary(self.saver.experiment_dir)
            self.writer = self.summary.create_summary()

        # PATH = args.path
        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.val_loader, self.nclass = make_data_loader(args, **kwargs)
        # self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)

        # Define network
        model = SCNN(nclass=self.nclass, backbone=args.backbone, output_stride=args.out_stride, cuda=args.cuda,
                     extension=args.ext)

        # Define Optimizer
        # optimizer = torch.optim.SGD(model.parameters(),args.lr, momentum=args.momentum,
        #                             weight_decay=args.weight_decay, nesterov=args.nesterov)
        optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

        # model, optimizer = amp.initialize(model,optimizer,opt_level="O1")

        # Define Criterion
        weight = None
        # criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)
        # self.criterion = SegmentationCELosses(weight=weight, cuda=args.cuda)
        # self.criterion = SegmentationCELosses(weight=weight, cuda=args.cuda)
        # self.criterion = FocalLoss(gamma=0, alpha=[0.2, 0.98], img_size=512*512)
        self.criterion1 = FocalLoss(gamma=5, alpha=[0.2, 0.98], img_size=512 * 512)
        self.criterion2 = disc_loss(delta_v=0.5, delta_d=3.0, param_var=1.0, param_dist=1.0,
                                    param_reg=0.001, EMBEDDING_FEATS_DIMS=21, image_shape=[512, 512])

        self.model, self.optimizer = model, optimizer

        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)
        # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                      args.epochs, len(self.val_loader), local_rank=args.local_rank)

        # Using cuda
        # if args.cuda:
        self.model = self.model.cuda()
        # if args.distributed:
        # self.model = DistributedDataParallel(self.model)
        # self.model = torch.nn.DataParallel(self.model)
            # patch_replication_callback(self.model)

        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            filename = 'checkpoint.pth.tar'
            args.resume = os.path.join(args.ckpt_dir, filename)
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            # if args.cuda:


            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v

            checkpoint['state_dict'] = new_state_dict

            self.model.load_state_dict(checkpoint['state_dict'])
            # else:
            # self.model.load_state_dict(checkpoint['state_dict'])
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

            loss = loss1 + 10 * loss2
            # loss = loss1
            output = output[1]
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

        if self.args.distributed:
            if self.args.local_rank == 0:
                if self.args.no_val:
                    # save checkpoint every epoch
                    is_best = False
                    self.saver.save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': self.model.module.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'best_pred': self.best_pred,
                    }, is_best)
            else:
                if self.args.no_val:
                    # save checkpoint every epoch
                    is_best = False
                    self.saver.save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': self.model.module.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'best_pred': self.best_pred,
                    }, is_best)

    def validation(self):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0

        destination_path = os.path.join(self.args.path,'seg_lane')
        if not os.path.isdir(destination_path):
            os.mkdir(destination_path,0o777)

        postprocessor = LanePostprocess.LaneNetPostProcessor()

        aa_sequence = mvpuai.MSequence()

        for i, sample in enumerate(tbar):
            # image, target = sample['image'], sample['label']
            image, lbl_path, resized_img = sample['image'], sample['lbl_path'], sample['resized_img']



            img = [image[0][...,_ind*512: (_ind+1)*512] for _ind in range(4)]
            img = np.stack(img+[resized_img[0]],axis=0)
            img = torch.from_numpy(img)
            if self.args.cuda:
                img = img.cuda()
                image = image.cuda()
            with torch.no_grad():
                output = self.model(img)

            pred = output[1]

            _upsampled=torch.nn.Upsample(size=[512,2048])
            overall_pred = pred[4,...].view([1,2,512,512])
            _upsampled=_upsampled(overall_pred)

            upsampled_final = torch.zeros(2,1024,2048)
            upsampled_final[:,512:,:512] = pred[0,...]
            upsampled_final[:, 512:, 512:1024] = pred[1, ...]
            upsampled_final[:, 512:, 1024:1024+512] = pred[2, ...]
            upsampled_final[:, 512:, 1024 + 512:2048] = pred[3, ...]
            upsampled_final = upsampled_final.view([1, 2, 1024, 2048])
            upsampled_final[..., 512:, :] = _upsampled
            upsampled_final = np.argmax(upsampled_final, axis=1)
            pred = upsampled_final.data.cpu().numpy()



            instance_seg = output[0]

            _upsampled_instance = torch.nn.Upsample(size=[512, 2048])
            overall_pred = instance_seg[4, ...].view([1, 21, 512, 512])
            _upsampled_instance = _upsampled_instance(overall_pred)

            upsampled_final_instance = torch.zeros(21, 1024, 2048)
            upsampled_final_instance[:, 512:, :512] = instance_seg[0, ...]
            upsampled_final_instance[:, 512:, 512:1024] = instance_seg[1, ...]
            upsampled_final_instance[:, 512:, 1024:1024 + 512] = instance_seg[2, ...]
            upsampled_final_instance[:, 512:, 1024 + 512:2048] = instance_seg[3, ...]
            upsampled_final_instance= upsampled_final_instance.view([1, 21, 1024, 2048])
            upsampled_final_instance[..., 512:, :] = _upsampled_instance

            instance_seg = upsampled_final_instance.data.cpu().numpy()
            # instance_seg = np.argmax(upsampled_final_instance, axis=1)






            # Add batch sample into evaluator

            # if i % 100 == 0:
            resized_img = np.squeeze(resized_img)
            pred = np.squeeze(pred)
            instance_seg = np.squeeze(instance_seg)

            # resized_img = np.transpose(resized_img.cpu().numpy(), (1, 2, 0))

            instance_seg = np.transpose(instance_seg, (1, 2, 0))

            postprocess_result = postprocessor.postprocess(
                binary_seg_result=pred,
                instance_seg_result=instance_seg,
                source_image=image
            )


            image = self.de_normalize(np.transpose(image[0].cpu().numpy(),(1,2,0)))

            # misc.imsave(destination_path + '/' + lbl_path[0],
            #             np.transpose(image.cpu().numpy(), (1, 2, 0)) + 3 * np.asarray(
            #                 np.stack((pred, pred, pred), axis=-1), dtype=np.uint8))

            show_source_image = np.zeros((1024, 2048, 3))

            show_source_image[512:, ...] = image
            image = show_source_image


            predicted_lanes = postprocess_result['lane_pts']

            # predicted_lanes = predicted_lanes[...,0]
            # bsp_lanes = []
            predicted_lanes = [np.asarray(pred_lane) for pred_lane in predicted_lanes]



            tensor_curvepts, tensor_cpts =inference(bsplineMat=predicted_lanes,i=i)

            tmp_mask = np.zeros(shape=(image.shape[0], image.shape[1]), dtype=np.uint8)
            src_lane_pts = np.asarray(tensor_curvepts)
            for lane_index, coords in enumerate(src_lane_pts):
                tmp_mask[tuple((np.int_(coords[:, 1]), np.int_(coords[:, 0])))] = lane_index + 1

            bsppts_mask = np.stack((tmp_mask, tmp_mask, tmp_mask), axis=-1)


            # misc.imsave(destination_path + '/mask_' + lbl_path[0],
            #             postprocess_result['mobis_mask_image'])
            # misc.imsave(destination_path + '/' + lbl_path[0],
            #             50*postprocess_result['mask_image']+50*postprocess_result['lanepts_mask'])
            misc.imsave(destination_path + '/' + lbl_path[0],
                                    postprocess_result['mobis_mask_image'])
            try:
                os.chmod(destination_path + '/'+ lbl_path[0],0o777)
            except:
                pass
            aa_sequence.add_frame(MFrame(i))
            for idx in range(tensor_cpts.shape[1]):
                _Obj = mvpuai.get_object_by_name(MString.Frame.Object.Type.LANE)
                _Obj.subclass_id = 1
                _Obj.instance_id = idx
                _list = []
                for ind in range(10):
                    _list.append(Point(int(tensor_cpts[0,idx,ind]), int(tensor_cpts[1,idx,ind])))

                _ctrl_pts = list([point.x, point.y] for point in _list)
                # b_spline = BSpline.Curve()
                # b_spline.degree = 4
                # b_spline.set_ctrlpts(_ctrl_pts)
                #
                # b_spline.knotvector = utilities.generate_knot_vector(b_spline.degree, len(_ctrl_pts))
                # b_spline.delta = 0.001
                # b_spline.evaluate()

                _cpts = []
                for _cpt in _ctrl_pts:
                    _cpts.append(_cpt[0])
                    _cpts.append(_cpt[1])

                _Obj.b_spline = _cpts


                aa_sequence.frame_list[-1].add_object(_Obj)
                # .add_frame(MFrame(0))




        self.write_json(aa_sequence)



    def de_normalize(self,img,mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):


        # img = np.array(img).astype(np.float32)
        img *= std
        img += mean

        img *= 255.0


        return img

    def write_json(self,aa_sequence):
        output_file_path = os.path.join(self.args.path,'json') + '/annotation_bs.json'
        mvpuai.write_json(output_file_path, aa_sequence)

        try:
            os.chmod(output_file_path, 0o777)
        except :
            pass



def main():
    parser = argparse.ArgumentParser(description="PyTorch SCNN Training")
    parser.add_argument('--distributed', type=bool, default=False,
                        help='backbone name (default: resnet)')
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    # parser.add_argument('--path', type=str, default='/mfc/data/compressed/Cityscapes/download',
    #                     help='path of cityscapes')

    parser.add_argument('--path', type=str, default='/mfc/data/mobis/real/30_aa_seg_test/1438_20190418_173931_DL/1438_20190418_173931_00000000',
                        help='path of LaneMobis')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='inference',
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
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15',
                        help='use which gpu to train, must be a \
                            comma-separated list of integers only (default=0,1)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=True,
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
    parser.add_argument('--single_gpu', default='0',
                        help='single_gpu(default: 0)')
    parser.add_argument('--ckpt_dir', default='/mfc/user/1623600/archive/Lane_final/run/LaneMobis/SCNN-resnet/experiment_15_False',
                        help='ckpt_dir')


    args = parser.parse_args()
    if args.distributed != True:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.single_gpu
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
        args.batch_size = 1
        args.test_batch_size = 1

    if args.checkname is None:
        args.checkname = 'SCNN-' + str(args.backbone)
    print(args)

    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    # if args.distributed and args.local_rank == 0:
    #     print('Starting Epoch:', trainer.args.start_epoch)
    #     print('Total Epoches:', trainer.args.epochs)
    # elif not args.distributed:
    #     print('Starting Epoch:', trainer.args.start_epoch)
    #     print('Total Epoches:', trainer.args.epochs)

    # for epoch in range(trainer.args.start_epoch, trainer.args.epochs):

    trainer.validation()

    trainer.writer.close()


if __name__ == "__main__":
    main()
