import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import os
print(os.getcwd())
import sys
import tensorflow as tf
import numpy as np
from BsplineModel.GetBspline import GetBspline_from_sampled_points
from BsplineModel.bspline_loss import BsplineLoss
from scipy import misc
from BsplineModel import BsplineModel
from tensorflow.keras.optimizers import Adam, SGD
from scipy import misc
import glob
import cv2
try:
    from BsplineModel.img_show import Points_to_Img, Points_to_bsp_curvepoints
except:
    pass

print(tf.__version__)

import mvpuai

from geomdl import utilities




# sys.path.insert(0,os.getcwd())
# sys.path.insert(0,os.getcwd()+'/util/Read_from_folder')
# sys.path.insert(0,os.getcwd()+'/util')
# sys.path.append('/mfc/user/1623600/MVPU1.0.road_information_extractor/AA_lane_API/')
# sys.path.append('/mfc/user/1623600/MVPU1.0.road_information_extractor/AA_lane_API/util')


from BsplineModel import Read_from_folder

showmode=False
eager_mode=False
pretrained = False
augment_mode = False


# if not showmode:
#     import horovod.tensorflow.keras as hvd
#     hvd.init()

nClass = 10
nIns = 20
Batch = 100#max 200
img_h = 1024
img_w = 2048
mode = {'TRAIN': 'TRAIN', 'VALIDATION': 'VALIDATION', 'TEST': 'TEST', 'SSG_TRAIN': 'SSG_TRAIN'}
name = '1'
# model_name = '-lane-best.'
model_name = '_ins.'
#img_show(value)
#To do list
# 1) Apply attention Maps : https://github.com/raghakot/keras-vis


def get_b_spline(ctrl_pts):
    #########
    ctrl_pts = mvpuai.reshape_coords_to_tuples(ctrl_pts)

    def get_degree_of_b_spline(num_of_ctrl_pts: int):
        if num_of_ctrl_pts >= 5:
            degree = 4
        elif num_of_ctrl_pts >= 1:
            degree = num_of_ctrl_pts - 1
        else:
            degree = 0

        return degree

    degree = get_degree_of_b_spline(len(ctrl_pts))
    knot_vector = utilities.generate_knot_vector(degree, len(ctrl_pts))
    start = knot_vector[degree]
    stop = knot_vector[-(degree + 1)]
    return utilities.linspace(start, stop, 101, decimals=6)


def get_bspline_with_knots(trainBs):
    ctrl_pts = [Bs.b_spline for Bs in trainBs]
    bspline_pts = [mvpuai.get_b_spline(cpts, 100) for cpts in ctrl_pts]

    # get knots
    knots = [get_b_spline(cpts) for cpts in ctrl_pts]

    bspline_pts = [np.asarray(bpts) for bpts in bspline_pts]
    bspline_ptsW = [bpts[:, 0] for bpts in bspline_pts]
    bspline_ptsH = [bpts[:, 1] for bpts in bspline_pts]

    bspline = [np.concatenate([np.expand_dims(ptsW, -1), np.expand_dims(ptsH, -1),np.expand_dims(knot, -1)], axis=-1) \
        for ptsW,ptsH,knot in zip(bspline_ptsW,bspline_ptsH,knots) ]

    return bspline

def train():
    train()

    os.environ["CUDA_VISIBLE_DEVICES"] = '4'  # use GPU with ID=0
    base_folder = "/mfc/data/mobis/real/40_ma/1438_20181211_114553/"
    # base_folder = "/mfc/user/1623600/.temp/"

    total_list = Read_from_folder.ParsingLainInsfromGT(Batch, base_folder, mode['TRAIN'])

    TrainImgList = total_list['train']
    ValImgList = total_list['val']

    #
    # next_train_dataset, next_val_dataset = \
    #     parsing_train_iterator(TrainImgList, ValImgList, img_h, img_w)  # Parsing iterator of dataset
    SavePath = '/mfc/user/1623600/MVPU1.0.road_information_extractor/AA_lane_API/Models/Bspline_model_simple/'
    if not pretrained:
        model = BsplineModel.BsplineModel(img_h, img_w, batch_size=Batch)
    else:
        model = BsplineModel.BsplineModel(img_h, img_w, batch_size=Batch)

        # saver = tf.python.train.Saver()
        # saver.restore()

        # model = tf.keras.models.load_model(SavePath+'saved_model.pb',compile=False)

    if not tf.__version__ == "1.13.0-rc1":
        # if False:
        optimizer = Adam(lr=0.001)  # (lr=0.001, decay=0.9)
        checkpoint_prefix = os.path.join(SavePath, 'ckpt')

        # latest_cpkt = tf.train.checkpoint_exists(checkpoint_prefix)
        step_counter = tf.compat.v1.train.get_or_create_global_step()
        checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer, step_counter=step_counter)
        status = checkpoint.restore(tf.train.latest_checkpoint(SavePath))
        # if latest_cpkt:
        #     print('Using latest checkpoint at ' + latest_cpkt)
        #     # Restore variables on creation if a checkpoint exists.
        #     checkpoint.restore(latest_cpkt)
        # else:
        #     print('failed to load checkpoints')

        # optimizer = SGD(lr=0.00001, decay=0.7)
        Epoch = 100000
        for e in range(Epoch):
            i = 0
            acc_loss = 0
            for trainImg, trainBs in zip(TrainImgList['img'], TrainImgList['bs']):
                # Img = misc.imread(trainImg)
                bsplineMat = get_bspline_with_knots(trainBs)
                smaplingvalues, bsplineMat_x, bsplineMat_y, tensor_sampling_inputs, tensor_basis, tensor_span = \
                    GetBspline_from_sampled_points(bsplineMat, img_w, img_h, Legacy=True)

                with tf.GradientTape() as tape:
                    smaplingvalues = tf.convert_to_tensor(smaplingvalues)
                    bsplineMat_x = tf.convert_to_tensor(bsplineMat_x)
                    bsplineMat_y = tf.convert_to_tensor(bsplineMat_y)
                    bsplineM = tf.stack((bsplineMat_x, bsplineMat_y))
                    # tensor_cpts = model(smaplingvalues,bsplineMat_x,bsplineMat_y)
                    tensor_cpts = model(bsplineM)
                    if pretrained:
                        model.load_weights(SavePath + '/my_model.h5')

                    loss_value, tensor_cpt_matrix_x = \
                        BsplineLoss(tensor_cpts, tensor_sampling_inputs,
                                    tensor_basis, tensor_span, DegreeOfCurve=4)

                grads = tape.gradient(loss_value, model.variables)
                optimizer.apply_gradients(zip(grads, model.variables))

                i += 1
                acc_loss += loss_value

                if i % 1 == 0:
                    print("current img : {:d}, loss:{:f}".format(i, acc_loss / (i)))
                    # print("tensor_cpts[0]",tensor_cpts[0][0])
                    # print("current img : {:d}, loss:{:f}".format(i, acc_loss/(i+1)/100))

                if i % 10 == 0:
                    print("tensor_cpts[0]", tensor_cpts[0][0])

                    # tf.saved_model.save(model, SavePath)
                    # model_path = os.path.join(SavePath, "model2.yaml")
                    # if os.path.isfile(model_path):
                    #     os.remove(model_path)
                    # with open(model_path, 'w') as fid:
                    #     model_yaml = model.to_yaml()
                    #     fid.write(model_yaml)
                    #     os.chmod(model_path, 0o777)
                    # tf.keras.models.save_model(model,SavePath + '/saved_model.h5')
                    # model.save_weights(SavePath + '/my_model.h5')
                    # model.save_weights(SavePath+'/my_model.h5')
                #     for dirpath, dirnames, filenames in os.walk(SavePath):
                #         for dirname in dirnames:
                #             path = os.path.join(dirpath, dirname)
                #             os.chmod(path, 0o777)  # for example
                #             for dir, _, fnames in os.walk(path):
                #                 for fname in fnames:
                #                     fname_path = os.path.join(path,fname)
                #                     os.chmod(fname_path, 0o777)
                #
                #         for filename in filenames:
                #             path = os.path.join(dirpath, filename)
                #             os.chmod(path, 0o777)

                if i % 500 == 499:  # validation
                    ######### predict curve points #########
                    bsplineMat = get_bspline_with_knots(trainBs)
                    smaplingvalues, bsplineMat_x, bsplineMat_y, tensor_sampling_inputs, tensor_basis, tensor_span \
                        = GetBspline_from_sampled_points(
                        bsplineMat, img_w, img_h)
                    smaplingvalues = tf.convert_to_tensor(smaplingvalues)
                    bsplineMat_x = tf.convert_to_tensor(bsplineMat_x)
                    bsplineMat_y = tf.convert_to_tensor(bsplineMat_y)
                    bsplineM = tf.stack((bsplineMat_x, bsplineMat_y))
                    # tensor_cpts = model(bsplineM)

                    # loss_value, tensor_cpt_matrix_x = \
                    #     BsplineLoss(tensor_cpts, tensor_sampling_inputs,
                    #                 tensor_basis, tensor_span, DegreeOfCurve=4)

                    tensor_cptss = model(bsplineM)
                    #######################################
                    img = np.zeros((img_h, img_w))
                    Points_to_Img(tensor_cptss, trainBs, img, i)
                    status.assert_consumed()
                    checkpoint.save(checkpoint_prefix)
                    print("check point saved at {:d}".format(i))
            status.assert_consumed()
            checkpoint.save(checkpoint_prefix)
            print("current epoch : {:d} is finished, loss:{:f}".format(e, acc_loss / (i + 1) / 100))
    else:
        Epoch = 100000
        for e in range(Epoch):
            i = 0
            acc_loss = 0
            for trainImg, trainBs in zip(TrainImgList['img'], TrainImgList['bs']):
                # Img = misc.imread(trainImg)
                bsplineMat = get_bspline_with_knots(trainBs)
                smaplingvalues, bsplineMat_x, bsplineMat_y, tensor_sampling_inputs, tensor_basis, tensor_span = \
                    GetBspline_from_sampled_points(bsplineMat, img_w, img_h)
                smaplingvalues = tf.convert_to_tensor(smaplingvalues)
                bsplineMat_x = tf.convert_to_tensor(bsplineMat_x)
                bsplineMat_y = tf.convert_to_tensor(bsplineMat_y)

                bsplineM = tf.stack((bsplineMat_x, bsplineMat_y))
                # tensor_cpts = model(smaplingvalues,bsplineMat_x,bsplineMat_y)
                # tensor_cpts = model(bsplineM)

                # tensor_cpts = model(smaplingvalues, bsplineMat_x, bsplineMat_y)
                loss_fn = BsplineLoss(tensor_cpts, tensor_sampling_inputs,
                                      tensor_basis, tensor_span, DegreeOfCurve=4)

                model = model(bsplineM)
                # BsplineLoss(tensor_cpts, tensor_sampling_inputs,
                #             tensor_basis, tensor_span, DegreeOfCurve=4)

                # loss_fn = {"emb_l": BsplineLoss(delta_v=0.5, delta_d=0.6, param_var=2, param_dist=1,
                #                                         param_reg=0.01)}

                # BsplineLoss(tensor_cpts, tensor_sampling_inputs, tensor_basis, tensor_span, DegreeOfCurve=4)

                model.compile(ptimizer=tf.train.RMSPropOptimizer(0.001),
                              loss=loss_fn,
                              metrics=['accuracy'])

def inference(bsplineMat,i=0,save_path = '/mfc/user/1623600/MVPU1.0.road_information_extractor/AA_lane_API/Models/Bspline_model_simple/'):
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)  # use GPU with ID=0

    SavePath = save_path

    model = BsplineModel.BsplineModel(img_h, img_w, batch_size=Batch)
    optimizer = Adam(lr=0.001)  # (lr=0.001, decay=0.9)
    checkpoint_prefix = os.path.join(SavePath, 'ckpt')

    # latest_cpkt = tf.train.checkpoint_exists(checkpoint_prefix)
    step_counter = tf.compat.v1.train.get_or_create_global_step()
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer, step_counter=step_counter)
    status = checkpoint.restore(tf.train.latest_checkpoint(SavePath))

    # smaplingvalues, bsplineMat_x, bsplineMat_y, tensor_sampling_inputs, tensor_basis, tensor_span = \
    #     GetBspline_from_sampled_points(bsplineMat, img_w, img_h, Legacy=True)

    smaplingvalues, bsplineMat_x, bsplineMat_y, tensor_sampling_inputs, tensor_basis, tensor_span = \
        GetBspline_from_sampled_points(bsplineMat, img_w, img_h)
    smaplingvalues = tf.convert_to_tensor(smaplingvalues)
    bsplineMat_x = tf.convert_to_tensor(bsplineMat_x)
    bsplineMat_y = tf.convert_to_tensor(bsplineMat_y)

    # tensor_cpts = model(smaplingvalues,bsplineMat_x,bsplineMat_y)
    # tensor_cpts = model(bsplineM)

    # model.load_weights(SavePath + '/my_model.h5')

    bsplineM = tf.stack((bsplineMat_x, bsplineMat_y))

    tensor_cptss = model(bsplineM)

    tensor_cptss = tensor_cptss.numpy()

    lanes_curvepts = []
    for ind in range(np.max(tensor_cptss.shape[1])):
        lanes_curvepts.append(Points_to_bsp_curvepoints(tensor_cptss[:,ind,:]))





    return lanes_curvepts, tensor_cptss
    #######################################
    # img = np.zeros((img_h, img_w))
    # Points_to_Img(img, tensor_cptss, i)
    # status.assert_consumed()
    # checkpoint.save(checkpoint_prefix)
    # print("check point saved at {:d}".format(i))





if __name__ == '__main__':
    train()
    # inference()







