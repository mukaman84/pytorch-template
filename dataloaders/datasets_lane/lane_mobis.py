import os
import numpy as np
import json
import scipy.misc as m
from PIL import Image
from torch.utils import data
from torchvision import transforms
from dataloaders import custom_transforms as tr

class LaneMobis(data.Dataset):
    NUM_CLASSES = 21

    def __init__(self, args, split="train"):

        self.root = args.path
        self.split = split
        self.args = args
        self.files = {}
        self.seg_files = {}
        self.json_data = {}
        self.dataset = {}

        if split == 'train':
            self.images_base = os.path.join(self.root, '100k')
            self.annotations_base = os.path.join(self.root, '100k')
            self.files[split] = []
            self.seg_files[split] = []
            self.json_data[split] = []
            self.dataset[split] = []
        elif split == 'val' or split == 'test':
            self.images_base = os.path.join(self.root, '100k')
            self.annotations_base = os.path.join(self.root, '100k')
            # self.images_base = os.path.join('/mfc/data/mobis/real', '40_ma')
            # self.annotations_base = os.path.join('/mfc/data/mobis/real', '40_ma')
            self.files[split] = []
            self.seg_files[split] = []
            self.json_data[split] = []
            self.dataset[split] = []
        else:
            raise NotImplementedError

        rec_list = os.listdir(self.annotations_base)
        seq_list = []
        for rec in rec_list:
            if split == 'val' or split == 'test':
                # if rec == '1438_20190214_133708' or rec == '1438_20190213_132328':
                if rec == '1438_20190228_140039':
                    _seq_list = os.listdir(os.path.join(self.annotations_base, rec))
                else:
                    _seq_list = []
            else:
                _seq_list = os.listdir(os.path.join(self.annotations_base, rec))
                # if rec == '1438_20181218_073134':
                #     _seq_list = os.listdir(os.path.join(self.annotations_base, rec))
                # else:
                #     _seq_list = []




            for seq in _seq_list:
                if not seq == '1438_20190213_132328_00079879'\
                        and not seq == '1438_20190213_132328_00069409'\
                        and not seq == '1438_20190213_132328_00069229' \
                        and not seq == '1438_20181218_073134_00053700':
                    if split == 'train':
                        # if seq == '1438_20181218_073134_00034200':
                        seq_list.append(os.path.join(self.annotations_base, rec, seq))
                    else:
                        seq_list.append(os.path.join(self.annotations_base, rec, seq))

        for seq_folder in seq_list:
            # print(seq_folder)
            if os.path.isdir(seq_folder):
                self.seg_recursive_glob(rootdir=seq_folder, suffix='.png',split=split)



            # self.files[split] = self.recursive_glob(rootdir=self.images_base, suffix='.png')

        self.void_classes = [-1]
        # self.valid_classes = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]
        self.valid_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        self.class_names = ['background','white_single_sold', 'white_single_dashed', 'white_double_solid', 'white_double_dashed',
                   'white_double_left_solid_right_dashed', 'white_double_left_dashed_right_solid', 'white_zigzag','white_botts_dot',
                   'yellow_single_sold', 'yellow_single_dashed', 'yellow_double_solid', 'yellow_double_dashed',
                   'yellow_double_left_solid_right_dashed', 'yellow_double_left_dashed_right_solid',
                   'blue_single_sold', 'blue_single_dashed', 'blue_double_solid', 'blue_double_dashed',
                   'blue_double_left_solid_right_dashed', 'blue_double_left_dashed_right_solid',
                   'unknown_single_sold', 'unknown_single_dashed', 'unknown_double_solid', 'unknown_double_dashed',
                   'unknown_double_left_solid_right_dashed', 'unknown_double_left_dashed_right_solid',
                   'removed_single_sold', 'removed_single_dashed', 'removed_double_solid', 'removed_double_dashed',
                   'removed_double_left_solid_right_dashed', 'removed_double_left_dashed_right_solid']

        self.ignore_index = 255
        self.class_map = dict(zip(self.valid_classes, range(self.NUM_CLASSES)))

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        print("Found %d %s images" % (len(self.files[split]), split))

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_path = self.dataset[self.split][index]['img'].rstrip()

        # img_path = self.files[self.split][index].rstrip()
        if self.split == 'train':
            # lbl_path = os.path.join(os.path.dirname(img_path)[:-3],'seg',os.path.basename(os.path.dirname(img_path)[:-4])+'_'+os.path.basename(img_path))
            lbl_path = os.path.join(os.path.dirname(img_path)[:-3], 'seg',os.path.basename(img_path))
            json_data = self.dataset[self.split][index]['json']['OBJECT_LIST']
        else:
            # lbl_path = os.path.join(os.path.dirname(img_path)[:-3], 'seg', os.path.basename(img_path))
            # lbl_path = os.path.join(os.path.dirname(img_path)[:-3], 'seg',os.path.basename(os.path.dirname(img_path)[:-4]) + '_' + os.path.basename(img_path))
            lbl_path = os.path.join(os.path.dirname(img_path)[:-3], 'seg',os.path.basename(img_path))
            json_data = self.dataset[self.split][index]['json']['OBJECT_LIST']
            # os.path.join(self.annotations_base,
            #                 img_path.split(os.sep)[-2],
            #                 os.path.basename(img_path)[:-15] + 'gtFine_labelIds.png')

        _img = Image.open(img_path).convert('RGB')
        _tmp = np.array(Image.open(lbl_path), dtype=np.uint8)


        #crop img to remove lines
        w, h = _img.size
        _img = _img.crop((0, int(h/2), w, h))
        _tmp = _tmp[512:,:]
        # _tmp = np.array(Image.open(lbl_path), dtype=np.uint8)

        # a = _tmp[...,1]
        # z = _tmp[...,0]
        _tmp_one= np.where(_tmp[...,0] != 80, np.zeros_like(_tmp[...,0]), 1)
        # _tmp_one_index = np.where(_tmp[..., 0] != 80)
        # _tmp_two = np.zeros_like(_tmp[...,1])
        _tmp_two = np.where(_tmp[...,0] == 80, _tmp[...,1], 0)



        if np.max(_tmp_two) >= 50:
            _tmp = np.subtract(_tmp_two,_tmp_one*49)
        else:
            _tmp = np.sum([_tmp_one, _tmp_two],axis=0)
        # b = _tmp[..., 1]


        # for object in json_data:
        #     if 'LANE' in object.keys():
        #         instance_ID = object['LANE']['INSTANCE_ID']
        #         seg_instance_ID = instance_ID + 1





        _tmp = self.encode_segmap(_tmp)
        _tmp_one = self.encode_segmap(_tmp_one)


        _target = Image.fromarray(_tmp.astype('uint8'))
        _target_bin = Image.fromarray(_tmp_one.astype('uint8'))

        sample = {'image': _img, 'label': _target, 'bin_label': _target_bin, 'lbl_path': lbl_path}

        if self.split == 'train':
            return self.transform_tr(sample)
        elif self.split == 'val':
            return self.transform_val(sample)
        elif self.split == 'test':
            return self.transform_ts(sample)

    def seg_recursive_glob(self,rootdir='.', suffix='.png',split='train'):
        if split=='train':
            seg_postfix = os.path.join(rootdir, 'seg')
            img_names = os.listdir(os.path.join(rootdir, 'img'))
            img_names =img_names[::10]
            # print(img_names)
            seg_names = sorted(os.listdir(seg_postfix))
            # seg_names = [name for name in seg_names if name[:-13] == os.path.basename(rootdir)]
            seg_base_names = [name for name in seg_names]
            json_path = os.path.join(rootdir, os.path.basename(rootdir) + '.json')
        else:
            seg_postfix = os.path.join(rootdir, 'seg')
            img_names = os.listdir(os.path.join(rootdir, 'img'))
            img_names = img_names[::10]
            seg_names = sorted(os.listdir(seg_postfix))
            # seg_names = [name for name in seg_names if name == os.path.basename(rootdir)]
            # seg_base_names = [name for name in seg_names]
            # seg_names = [name for name in seg_names if name[:-13] == os.path.basename(rootdir)]
            seg_base_names = [name for name in seg_names]
            json_path = os.path.join(rootdir, os.path.basename(rootdir) + '.json')


        # seg_names = [os.path.join(seg_postfix, name) for name in seg_names]
        seg_names.sort()

        json_data = json.load(open(json_path, 'r'))

        dataset = dict()


        for frame in json_data['FRAME_LIST']:
            if '{:08d}.png'.format(frame['FRAME_METADATA']['NUMBER']) in seg_base_names:
                dataset['img'] = os.path.join(rootdir, 'img','{:08d}.png'.format(frame['FRAME_METADATA']['NUMBER']))
                dataset['seg'] = os.path.join(seg_postfix, os.path.basename(rootdir+'_{:08d}.png'.format(frame['FRAME_METADATA']['NUMBER'])))
                dataset['json'] = frame
                self.dataset[split].append(dataset)





        self.files[split] += [os.path.join(rootdir, 'img',name) for name in img_names if name in seg_base_names]
        self.seg_files[split] += seg_names

        # print(self.files[split])
        # self.seg_files[split] += seg_names

    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask

    def recursive_glob(self, rootdir='.', suffix=''):
        """Performs recursive glob with given suffix and rootdir
            :param rootdir is the root directory
            :param suffix is the suffix to be searched
        """
        return [os.path.join(looproot, filename)
                for looproot, _, filenames in os.walk(rootdir)
                for filename in filenames if filename.endswith(suffix)]

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
            tr.RandomRotate(degree=90),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_ts(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixedResize(size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

if __name__ == '__main__':
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 513
    args.crop_size = 513

    cityscapes_train = CityscapesSegmentation(args, split='train')

    dataloader = DataLoader(cityscapes_train, batch_size=2, shuffle=True, num_workers=2)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='cityscapes')
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        if ii == 1:
            break

    plt.show(block=True)