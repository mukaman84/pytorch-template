import scipy
import tensorflow as tf
import numpy as np
import os
import mvpuai
from scipy import misc
from BsplineModel.visualization import add_color

def img_show(next_train_dataset,bin_flag=True):
    iterator = next_train_dataset.make_initializable_iterator()
    next_elements = iterator.get_next()
    with tf.Session() as sess:
        sess.run(iterator.initializer)
        while(next_elements):
            a = sess.run(next_elements)
            # img_show(a[0][1])
            scipy.misc.imsave('/mfc/img_pcd_pre/archive/remote_rep/Dontcare/keras_multi_gpu/log/outfile.png', (a[0][0]+0.5)*255)
            # b= np.where((np.round(a[0][0]+0.5))==1.)

            if bin_flag:
                # scipy.misc.imsave('/mfc/img_pcd_pre/archive/remote_rep/Dontcare/keras_multi_gpu/log/lane_bin.png', a[1][0][0][..., 0] * 255)
                scipy.misc.imsave('/mfc/img_pcd_pre/archive/remote_rep/Dontcare/keras_multi_gpu/log/lane_bin.png',
                                  a[1][0][..., 1] * 255)
                for ind in range(a[1][1].shape[-1]):
                    if ind == 0:
                        lane_cls = np.zeros_like(a[1][1][..., ind])
                    else:
                        lane_cls += a[1][1][..., ind]*ind

                scipy.misc.imsave('/mfc/img_pcd_pre/archive/remote_rep/Dontcare/keras_multi_gpu/log/lane_cls.png',
                                  lane_cls[0])
                lane_ins = a[1][2][0][..., 0]
                scipy.misc.imsave(
                    '/mfc/img_pcd_pre/archive/remote_rep/Dontcare/keras_multi_gpu/log/lane_instance.png',
                    lane_ins * 20)
                # scipy.misc.imsave('/mfc/img_pcd_pre/archive/remote_rep/Dontcare/keras_multi_gpu/log/ground.png',a[1][1][..., 0])



                os.chmod('/mfc/img_pcd_pre/archive/remote_rep/Dontcare/keras_multi_gpu/log/lane_cls.png', 0o777)
                os.chmod('/mfc/img_pcd_pre/archive/remote_rep/Dontcare/keras_multi_gpu/log/lane_instance.png',
                         0o777)
            else:
                scipy.misc.imsave('/mfc/img_pcd_pre/archive/remote_rep/Dontcare/keras_multi_gpu/log/lane_bin.png',
                                  a[1][0][..., 0] * 255)

            os.chmod('/mfc/img_pcd_pre/archive/remote_rep/Dontcare/keras_multi_gpu/log/outfile.png', 0o777)
            os.chmod('/mfc/img_pcd_pre/archive/remote_rep/Dontcare/keras_multi_gpu/log/lane_bin.png', 0o777)

            # print("111")
            next_elements = iterator.get_next()


def Save_Img(img,index,Points,color,tag):
    Points = [np.round(be) for be in Points]
    for be in Points:
        for epd in range(10):
            try:
                img[int(be[1]) + epd, int(be[0]) + epd] = int(color)
            except IndexError as e:
                print(e)

    img = add_color(img)
    misc.imsave("/mfc/user/1623600/.temp2/line_{:05d}_{:s}.png".format(index,tag), img)
    os.chmod('/mfc/user/1623600/.temp2/line_{:05d}_{:s}.png'.format(index,tag), 0o777)



def Points_to_Img(tensor_cpt, trainBs,img,index=0):
    tensor_cpt_x = tensor_cpt[0][0]
    tensor_cpt_y = tensor_cpt[1][0]
    cpts = []

    for cpt_x, cpt_y in zip(tensor_cpt_x, tensor_cpt_y):
        cpts.append(int(cpt_x))
        cpts.append(int(cpt_y))


    # print("cpt_x",cpt_x)
    # print("cpt_y", cpt_y)
    # a = mvpuai.get_b_spline(cpts, 100)
    # b = mvpuai.get_b_spline(trainBs[0].b_spline, 100)
    return mvpuai.get_b_spline(cpts, 100)
    # Save_Img(img, index,a,1,'predict')
    # # img = np.zeros_like(img)
    # Save_Img(img, index, b, 2, 'gt')

def Points_to_bsp_curvepoints(tensor_cpt):
    tensor_cpt_x = tensor_cpt[0]
    tensor_cpt_y = tensor_cpt[1]
    cpts = []

    for cpt_x, cpt_y in zip(tensor_cpt_x, tensor_cpt_y):
        cpts.append(int(cpt_x))
        cpts.append(int(cpt_y))


    print("cpt_x",cpt_x)
    print("cpt_y", cpt_y)
    # a = mvpuai.get_b_spline(cpts, 100)
    # b = mvpuai.get_b_spline(trainBs[0].b_spline, 100)
    return mvpuai.get_b_spline(cpts, 100)
    # Save_Img(img, index,a,1,'predict')
    # # img = np.zeros_like(img)
    # Save_Img(img, index, b, 2, 'gt')