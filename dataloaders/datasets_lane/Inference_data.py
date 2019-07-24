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

        if split == 'val' or split == 'test':
            self.images_base = self.root
            self.annotations_base = os.path.join(self.root, 'img')
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


        self.seg_recursive_glob(rootdir=self.images_base, suffix='.png',split=split)



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
        img_path = self.dataset[self.split][0]['img'][index].rstrip()

        # img_path = self.files[self.split][index].rstrip()
        if self.split == 'train':
            # lbl_path = os.path.join(os.path.dirname(img_path)[:-3],'seg',os.path.basename(os.path.dirname(img_path)[:-4])+'_'+os.path.basename(img_path))
            lbl_path = os.path.basename(img_path)

        else:
            # lbl_path = os.path.join(os.path.dirname(img_path)[:-3], 'seg', os.path.basename(img_path))
            # lbl_path = os.path.join(os.path.dirname(img_path)[:-3], 'seg',os.path.basename(os.path.dirname(img_path)[:-4]) + '_' + os.path.basename(img_path))
            lbl_path = os.path.basename(img_path)

            # os.path.join(self.annotations_base,
            #                 img_path.split(os.sep)[-2],
            #                 os.path.basename(img_path)[:-15] + 'gtFine_labelIds.png')

        _img = Image.open(img_path).convert('RGB')
        # _tmp = np.array(Image.open(lbl_path), dtype=np.uint8)


        #crop img to remove lines
        w, h = _img.size
        _img = _img.crop((0, int(h/2), w, h))
        # _tmp = _tmp[512:,:]
        # _tmp = np.array(Image.open(lbl_path), dtype=np.uint8)

        # a = _tmp[...,1]
        # z = _tmp[...,0]
        # _tmp_one= np.where(_tmp[...,0] != 80, np.zeros_like(_tmp[...,0]), 1)
        # _tmp_one_index = np.where(_tmp[..., 0] != 80)
        # _tmp_two = np.zeros_like(_tmp[...,1])
        # _tmp_two = np.where(_tmp[...,0] == 80, _tmp[...,1], 0)



        # if np.max(_tmp_two) >= 50:
        #     _tmp = np.subtract(_tmp_two,_tmp_one*49)
        # else:
        #     _tmp = np.sum([_tmp_one, _tmp_two],axis=0)
        # b = _tmp[..., 1]


        # for object in json_data:
        #     if 'LANE' in object.keys():
        #         instance_ID = object['LANE']['INSTANCE_ID']
        #         seg_instance_ID = instance_ID + 1





        # _tmp = self.encode_segmap(_tmp)
        # _tmp_one = self.encode_segmap(_tmp_one)

        _lbl_path = lbl_path
        sample = {'image': _img, 'lbl_path': _lbl_path}


        return self.transform_val(sample)

    def seg_recursive_glob(self,rootdir='.', suffix='.png',split='train'):


        img_names = os.listdir(os.path.join(rootdir, 'img'))


        # seg_names = [name for name in seg_names if name == os.path.basename(rootdir)]
        # seg_base_names = [name for name in seg_names]
        # seg_names = [name for name in seg_names if name[:-13] == os.path.basename(rootdir)]

                # seg_names = [os.path.join(seg_postfix, name) for name in seg_names]
        img_names.sort()


        dataset = dict()


        for _, name in enumerate(img_names):
            if _ != 0:
                dataset['img'].append(os.path.join(rootdir, 'img',name))
            else:
                dataset['img'] = [os.path.join(rootdir, 'img', name)]





        self.dataset[split].append(dataset)





        self.files[split] += [os.path.join(rootdir, 'img',name) for name in img_names]


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

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.Inf_FixScaleCrop(crop_size=self.args.crop_size),
            tr.Inf_Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.Inf_ToTensor()])

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