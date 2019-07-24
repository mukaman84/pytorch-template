"""
Copyright 2018 The Mobis AA team. All Rights Reserved.
======================================
base scenario data parsing API
======================================
Author : Dongyul Lee
Issue date : 17, Oct, 2018
ver : 1.0.0

============
Descriptions
============
data parsing interface

============
depedencies
============
tensorflow=1.12
=====

"""
import os
import tensorflow as tf
import numpy as np
import mvpuai
# import cv2
# from skimage import io
from tensorflow.keras import backend as K

print(os.getcwd())


from config.load_save_config import *

import h5py
import scipy.misc as misc
from PIL import Image
import cv2
from BsplineModel.SaveImg import ImgSave
from BsplineModel.CropLaneMask import CropInsLane



def _list_from_file(path, sub=None, seg=None, mode=None, lane = False, ssg=False, ma_seg=False):
    """
    :param
    sub : subfolder name
    seg : segmentation folder name
    mode : train or test phase
    lane : bool variable to indicate that this folder will be used for lane detection

    :return:
    """
    if not lane:
        with open(path + sub + '/Segmentation/' + mode, 'r') as fid:
            lines = fid.read().split('\n')[:-1]
            train_image_list = []
            train_label_list = []
            for _, file_name in enumerate(lines):
                image_name = file_name + '.jpg'
                label_name = file_name + '.png'
                train_image_list.append(path + 'JPEGImages/' + image_name)
                train_label_list.append(os.path.join(os.getcwd(), 'data',seg,label_name))



    if lane and ssg:
        sub_folder_list = os.listdir(path)

        ###generate sequence folder list
        folder_list = []
        img_list = []
        label_binary_list = []
        label_class_list = []
        label_instance_list = []
        for fld in sub_folder_list:
            if fld[0:3] == 'seq':
                # generate total file list
                folder_list.append(fld)

        _img_list, _label_binary_list, _label_class_list, _label_instance_list = __extract_train_val_gt_list(
            path, folder_list,ssg)
        img_list += _img_list
        label_binary_list += _label_binary_list
        label_class_list += _label_class_list
        label_instance_list += _label_instance_list

        train_img_list=[]
        train_label_binary_list =[]
        train_label_class_list =[]
        train_label_instance_list = []
        val_img_list=[]
        val_label_binary_list=[]
        val_label_class_list=[]
        val_label_instance_list = []
        for ind,_ in enumerate(img_list):
            if ind % 4 == 0:
                val_img_list.append(img_list[ind])
                val_label_binary_list.append(label_binary_list[ind])
                val_label_class_list.append(label_class_list[ind])
                val_label_instance_list.append(label_instance_list[ind])
            else:
                train_img_list.append(img_list[ind])
                train_label_binary_list.append(label_binary_list[ind])
                train_label_class_list.append(label_class_list[ind])
                train_label_instance_list.append(label_instance_list[ind])

        gt = {}
        gt['train'] = {}
        gt['val'] = {}

        gt['train']['img'] = train_img_list
        gt['train']['bin'] = train_label_binary_list
        gt['train']['ins'] = train_label_class_list
        gt['train']['cls'] = train_label_instance_list

        gt['val']['img'] = val_img_list
        gt['val']['bin'] = val_label_binary_list
        gt['val']['ins'] = val_label_class_list
        gt['val']['cls'] = val_label_instance_list

        return gt

    elif not lane and not ssg and not ma_seg:
        "Searching lane img and gt directories under MA folder without sequences folder"
        folder_list = os.listdir(path)

        if True:
            train_folder_list = folder_list
            validation_folder_list = folder_list[-2:]
        else:
            train_folder_list = folder_list[0:-2]
            validation_folder_list = folder_list[-2:]

        train_img_list, train_label_binary_list, train_label_class_list, train_label_instance_list = __extract_train_val_gt_list(path,train_folder_list)
        val_img_list, val_label_binary_list, val_label_class_list, val_label_instance_list = __extract_train_val_gt_list(
            path, validation_folder_list)

        gt = {}
        gt['train'] = {}
        gt['val'] = {}

        gt['train']['img'] = train_img_list
        gt['train']['bin'] = train_label_binary_list
        gt['train']['ins'] = train_label_class_list
        gt['train']['cls'] = train_label_instance_list

        gt['val']['img'] = val_img_list
        gt['val']['bin'] = val_label_binary_list
        gt['val']['ins'] = val_label_class_list
        gt['val']['cls'] = val_label_instance_list

        return gt

    elif ma_seg:
        "Searching for segmentation images in the scoring DB. This function is temporally used in Q4, 2018"
        folder_list = os.listdir(path)
        temp_folder_list = []
        for folder in folder_list:
            if not folder[0:4] == 'hwas' and not folder[0:4] == 'jukj':
                temp_folder = os.listdir(os.path.join(path, folder))
                SSG = True
                if SSG:
                    for fld in temp_folder:
                            child_folders = os.listdir(os.path.join(path, folder, fld))
                            for child in child_folders:
                                temp_folder_list.append(os.path.join(path, folder, fld, child))

                else:
                    for fld in temp_folder:
                        temp_folder_list.append(os.path.join(path,folder,fld))

        folder_list = temp_folder_list

        if True:
            train_folder_list = folder_list
            validation_folder_list = folder_list[-50:]
        else:
            train_folder_list = folder_list[0:-2]
            validation_folder_list = folder_list[-2:]




        train_img_list, train_label_binary_list, train_label_class_list, train_label_instance_list = __extract_train_val_gt_list(path,train_folder_list,ma_seg=True)
        val_img_list, val_label_binary_list, val_label_class_list, val_label_instance_list = __extract_train_val_gt_list(
            path, validation_folder_list,ma_seg=True)

        gt = {}
        gt['train'] = {}
        gt['val'] = {}

        gt['train']['img'] = train_img_list
        gt['train']['bin'] = train_label_binary_list
        gt['train']['ins'] = train_label_class_list
        gt['train']['cls'] = train_label_instance_list

        gt['val']['img'] = val_img_list
        gt['val']['bin'] = val_label_binary_list
        gt['val']['ins'] = val_label_class_list
        gt['val']['cls'] = val_label_instance_list

        return gt



def __extract_train_val_gt_list(path,folder_list,ssg=False,ma_seg=False):

    gt_img_list = []
    # if not ma_seg:
    gt_label_binary_list = []
    gt_label_instance_list = []
    gt_label_class_list = []

    if not ma_seg:
        for fld in folder_list:

            temp_img_list, temp_bin_list, temp_ins_list, temp_cls_list = __extract_gt_list(path=path,fld=fld,ssg=ssg)
            gt_img_list += temp_img_list
            gt_label_binary_list += temp_bin_list
            gt_label_instance_list += temp_ins_list
            gt_label_class_list += temp_cls_list

        return gt_img_list, gt_label_binary_list, gt_label_class_list, gt_label_instance_list
    else:
        for fld in folder_list:
            temp_img_list, temp_cls_list = __ma_seg_extract_gt_list(path=path, fld=fld, ssg=ssg)
            gt_img_list += temp_img_list
            gt_label_class_list += temp_cls_list

        return gt_img_list, gt_label_binary_list, gt_label_class_list, gt_label_instance_list

def __ma_seg_extract_gt_list(path,fld=None,ssg=False):
    img_list = []
    gt_bin_list = []
    gt_instance_list = []
    gt_seg_list = []
    # dataset_dir = os.path.join(path, fld)
    dataset_dir = fld

    temp_png_list = glob.glob(dataset_dir + '/img/' + '*.png')
    temp_png_list.sort()
    for temp_index in temp_png_list:
        temp_number = os.path.basename(temp_index)[:-4]
        # temp_number = temp_number.split('_')[0]

        if os.path.isfile(dataset_dir + '/img/' + temp_number + '.png'):
            img_list.append(dataset_dir + '/img/' + temp_number + '.png')
            gt_seg_list.append(dataset_dir + '/seg/' + temp_number + '.png')

    return img_list, gt_seg_list

def __extract_gt_list(path,fld=None,ssg=False):
    img_list = []
    gt_bin_list = []
    gt_instance_list = []
    gt_seg_list = []


    if not ssg:
        mask_folder = '/mask/'
        dataset_dir = os.path.join(path, fld)
        temp_dedium_path = '/'
    else:
        mask_folder = '/mask/' + fld + '/'
        dataset_dir = path
        temp_dedium_path = '/' + fld + '/'

    temp_png_list = glob.glob(dataset_dir + mask_folder + '*_{:s}.png'.format('bin'))
    temp_png_list.sort()
    for temp_index in temp_png_list:
        temp_number = os.path.basename(temp_index)[:-4]
        temp_number = temp_number.split('_')[0]

        if os.path.isfile(dataset_dir + temp_dedium_path + 'png/' + temp_number + '.png'):
            img_list.append(dataset_dir + temp_dedium_path + 'png/' + temp_number + '.png')
            gt_bin_list.append(dataset_dir + mask_folder + temp_number + '_bin.png')
            gt_instance_list.append(dataset_dir + mask_folder + temp_number + '_instance.png')
            gt_seg_list.append(dataset_dir + mask_folder + temp_number + '_seg.png')

        a=dataset_dir + temp_dedium_path + 'img/' + temp_number + '.png'
        if os.path.isfile(dataset_dir + temp_dedium_path + 'img/' + temp_number + '.png'):
            img_list.append(dataset_dir + temp_dedium_path + 'img/' + temp_number + '.png')
            gt_bin_list.append(dataset_dir + mask_folder + temp_number + '_bin.png')
            gt_instance_list.append(dataset_dir + mask_folder + temp_number + '_instance.png')
            gt_seg_list.append(dataset_dir + mask_folder + temp_number + '_seg.png')





    return img_list, gt_bin_list, gt_instance_list, gt_seg_list



def parsing_imglbl_list(mode):
    args = get_config_args(mode)
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.object_mode=='pascal':

        # MAIN_PATH = '/mfc/data/compressed/PASCAL-VOC/download/2012/VOCdevkit/VOC2012/'
        sub_folders = {'gt': 'Annotations', 'list': 'ImageSets', 'ori': 'JPEGImages', 'seg': 'SegmentationClassRaw',
                       'ins': 'SegmentationObjec'}

        read_mode = {'TRAIN': 'train.txt', 'VALIDATION': 'val.txt'}
        # Read train img list
        train_image_list, train_label_list = _list_from_file(args.base_dir, sub_folders['list'], sub_folders['seg'],read_mode['TRAIN'])
        # Read test img list
        test_image_list, test_label_list = _list_from_file(args.base_dir, sub_folders['list'], sub_folders['seg'],read_mode['VALIDATION'])
        return args, (train_image_list, train_label_list, test_image_list, test_label_list)

    elif args.object_mode=='lane':
        return args, _list_from_file(path=args.base_dir, lane=True)
    elif args.object_mode=='ssg':
        return args, _list_from_file(path=args.base_dir, lane=True, ssg=True)
    elif args.object_mode=='ma_seg':
        return args, _list_from_file(path=args.base_dir, lane=True, ma_seg=True)
    else:
        return args, _list_from_file(path=args.base_dir, lane=True)


def ImgSave(aSegFile,loadedSegImg,LaneId=''):
    aSegFile = aSegFile.split('.')[0]
    if LaneId=='':
        SegName = "/mfc/user/1623600/.temp/{:s}.png".format(os.path.basename(aSegFile))
    else:
        SegName = "/mfc/user/1623600/.temp/{:s}_{:s}.png".format(os.path.basename(aSegFile), str(LaneId))

    misc.imsave(SegName, loadedSegImg)
    try:
        os.chmod(SegName, 0o777)
    except:
        print("Permission denied")




def ParsingLaineIns(aSegFile,OveralSegImg):
    NofLane = np.max(OveralSegImg[...,1])
    for LaneInstance in range(NofLane+1):
        InsSegImg = np.zeros_like(OveralSegImg)
        LenValidInsN = len(np.where(OveralSegImg[..., 1] == LaneInstance)[0])

        LaneCondition = (OveralSegImg[..., 0] == 80)
        InsCondition = (OveralSegImg[..., 1] == LaneInstance)
        LaneInsCondition = np.multiply(LaneCondition, InsCondition)

        if LenValidInsN and np.max(LaneInsCondition):
            InsSegImg[...,0] = np.where(np.invert(LaneInsCondition),
                                        InsSegImg[..., 0], 255)
            InsSegImg[...,1] = np.where(np.invert(LaneInsCondition),
                                 InsSegImg[..., 1], 255)
            CropInsLane(aSegFile, InsSegImg, LaneId=LaneInstance)


            # ImgSave(aSegFile, InsSegImg, LaneId=LaneInstance)




def LoadSegFile(aSegFile):
    OveralSegImg=misc.imread(aSegFile)
    LainID = 80
    LaneSegImg = np.zeros_like(OveralSegImg)
    OveralSegImg[..., 2] = LaneSegImg[..., 2]
    LaneSegImg=np.where(OveralSegImg[...,0]!=LainID,LaneSegImg[...,0],LainID)
    OveralSegImg[..., 0] = LaneSegImg
    LaneSegImg = np.where(OveralSegImg[..., 0] == LainID, OveralSegImg[..., 1], 0)
    OveralSegImg[..., 1] = LaneSegImg
    ImgSave(aSegFile, OveralSegImg)

    ParsingLaineIns(aSegFile,OveralSegImg)


def ExtractImgList(FolderList):
    ImgList = []


    for Folder in FolderList:
        ImgList += glob.glob(Folder + '/*.png')
    return img_list, gt_seg_list

def LoadjsonFile(aJsonFile, mImages_list, mObjecs_list):
    if os.path.isfile(aJsonFile[0]):
        print(aJsonFile[0])
        mSeg = mvpuai.read_json(aJsonFile[0])
        crop_sequence = mvpuai.MSequence()

        for frame in range(mSeg.meta.num_of_frames):
            # if mSeg.frame_list[frame].meta.stage >= 2:
            # iteration on entire object and check whether lane or road boundary
            for obj_idx in range(len(mSeg.frame_list[frame].object_list)):
                ########## Process only for ROAD_BOUNDARY ##########
                if mSeg.frame_list[frame].object_list[obj_idx].NAME == 'ROAD_BOUNDARY' or \
                    mSeg.frame_list[frame].object_list[obj_idx].NAME == 'LANE' :
                    mImages_list.append(os.path.dirname(aJsonFile[0]) + '/img/' + str(mSeg.frame_list[frame].meta.num).rjust(8,'0') + '.png')
                    mObjecs_list.append(mSeg.frame_list[frame].object_list[obj_idx])





        return mImages_list, mObjecs_list







def LoadFolderList(Batch,Mainfolder):
    RecordFolders = os.listdir(Mainfolder)
    RecordFolders.sort()

    folder_list = []

    mImages_list = []
    mObjecs_list = []

    for recordfolder in RecordFolders:
        if not recordfolder =='ssg' and os.path.isdir(Mainfolder + recordfolder) and \
                (recordfolder[-8:-4] == '0000' or recordfolder[-5] == '1' or recordfolder[-5] == '2' or recordfolder[-5] == '3'):
            SeqFolders = os.listdir(Mainfolder + recordfolder)
            SeqFolders.sort()
            JSON = True
            if not JSON:
                for seqfolder in SeqFolders:
                    if os.path.isdir(Mainfolder + '/' + recordfolder + '/' + seqfolder):
                        SegFolder = Mainfolder + '/' + recordfolder + '/' + seqfolder + '/seg/'
                        folder_list.append(SegFolder)
                        SegFiles=glob.glob(SegFolder + '*.png')
                        SegFiles.sort()
                        for aSegFile in SegFiles:
                            LoadSegFile(aSegFile)
            else:
                # if
                SegFolder = Mainfolder + '/' + recordfolder + '/'
                SegFiles = glob.glob(SegFolder + '*.json')
                temp_mImages_list, temp_mObjecs_list = LoadjsonFile(SegFiles, mImages_list, mObjecs_list)

                mImages_list += temp_mImages_list
                mObjecs_list += temp_mObjecs_list

    Images_list = []
    Objecs_list = []


    mod =0


    for img, obj in zip(mImages_list, mObjecs_list):
        if mod == 0:
            temp_mImages_list=[img]
            temp_mObjecs_list=[obj]
            mod += 1
            if Batch == 1:
                mod =0
                Images_list.append(temp_mImages_list)
                Objecs_list.append(temp_mObjecs_list)

        elif Batch >1:
            if mod % (Batch-1) == 0:
                mod = 0
                temp_mImages_list.append(img)
                temp_mObjecs_list.append(obj)
                Images_list.append(temp_mImages_list)
                Objecs_list.append(temp_mObjecs_list)
            else:
                mod += 1
                temp_mImages_list.append(img)
                temp_mObjecs_list.append(obj)




    gt = {}
    gt['train'] = {}
    gt['val'] = {}

    gt['train']['img'] = Images_list
    gt['train']['bs'] = Objecs_list


    gt['val']['img'] = Images_list
    gt['val']['bs'] = Objecs_list

    return gt


def LoadFolderList_processed(Mainfolder):
    LaneFiles = glob.glob(Mainfolder + '*.png')
    PngList = []

    for pngfile in LaneFiles:
        if pngfile[-6] =='_' or pngfile[-7] =='_':
            PngList.append(pngfile)

            # LoadSegFile(aSegFile)

    train_folder_list = PngList
    validation_folder_list = PngList[-200:]

    gt = {}
    gt['train'] = train_folder_list
    gt['val'] = validation_folder_list

    return gt
    # gt['train']['img'] = train_folder_list
    # gt['train']['bin'] = validation_folder_list

def ParsingLainInsfromGT(Batch,Mainfolder,mode):
    # return LoadFolderList_processed(Mainfolder)
    return LoadFolderList(Batch,Mainfolder)






if __name__ == '__main__':
    ###################################################################
    # Data Preparation
    ###################################################################
    pass

