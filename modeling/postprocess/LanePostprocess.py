import os.path as ops
import math

import cv2
import glog as log
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from modeling.postprocess import getlane
import scipy.misc as misc
import os

def _connect_components_analysis(image):
    """
    connect components analysis to remove the small components
    :param image:
    :return:
    """
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    return cv2.connectedComponentsWithStats(gray_image, connectivity=8, ltype=cv2.CV_32S)


def _morphological_process(image, kernel_size=5):
    """
    morphological process to fill the hole in the binary segmentation result
    :param image:
    :param kernel_size:
    :return:
    """
    if len(image.shape) == 3:
        raise ValueError('Binary segmentation result image should be a single channel image')

    if image.dtype is not np.uint8:
        image = np.array(image, np.uint8)

    kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(kernel_size, kernel_size))

    # close operation fille hole
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=1)

    return closing


class _LaneNetCluster(object):
    """
     Instance segmentation result cluster
    """

    def __init__(self):
        """

        """
        self._color_map = [np.array([255, 0, 0]),
                           np.array([0, 255, 0]),
                           np.array([0, 0, 255]),
                           np.array([125, 125, 0]),
                           np.array([0, 125, 125]),
                           np.array([125, 0, 125]),
                           np.array([50, 100, 50]),
                           np.array([70, 50, 70]),
                           np.array([125, 0, 125]),
                           np.array([50, 50, 50]),
                           np.array([500, 150, 200])]

        self._mobis_map = [np.array([80, 0, 0]),
                           np.array([80, 1, 0]),
                           np.array([80, 2, 0]),
                           np.array([80, 3, 0]),
                           np.array([80, 4, 0]),
                           np.array([80, 5, 0]),
                           np.array([80, 6, 0]),
                           np.array([80, 7, 0]),
                           np.array([80, 8, 0]),
                           np.array([80, 9, 0]),
                           np.array([80, 10, 0]),
                           np.array([80, 11, 0]),
                           np.array([80, 12, 0]),
                           np.array([80, 13, 0]),
                           np.array([80, 14, 0]),
                           np.array([80, 15, 0]),
                           np.array([80, 16, 0]),
                           np.array([80, 17, 0]),
                           np.array([80, 18, 0]),
                           np.array([80, 19, 0]),
                           np.array([80, 20, 0]),
                           np.array([80, 21, 0]),
                           np.array([80, 22, 0]),
                           np.array([80, 23, 0]),
                           np.array([80, 24, 0]),
                           np.array([80, 25, 0]),
                           np.array([80, 26, 0]),
                           np.array([80, 27, 0]),
                           np.array([80, 28, 0])]


    @staticmethod
    def _embedding_feats_dbscan_cluster(embedding_image_feats,eps=0.35, min_samples=1000):
        """
        dbscan cluster
        :param embedding_image_feats:
        :return:
        """
        db = DBSCAN(eps=eps, min_samples=min_samples)
        try:
            features = StandardScaler().fit_transform(embedding_image_feats)
            db.fit(features)
        except Exception as err:
            log.error(err)
            ret = {
                'origin_features': None,
                'cluster_nums': 0,
                'db_labels': None,
                'unique_labels': None,
                'cluster_center': None
            }
            return ret
        db_labels = db.labels_
        unique_labels = np.unique(db_labels)

        num_clusters = len(unique_labels)
        cluster_centers = db.components_

        ret = {
            'origin_features': features,
            'cluster_nums': num_clusters,
            'db_labels': db_labels,
            'unique_labels': unique_labels,
            'cluster_center': cluster_centers
        }

        return ret

    @staticmethod
    def _get_lane_embedding_feats(binary_seg_ret, instance_seg_ret):
        """
        get lane embedding features according the binary seg result
        :param binary_seg_ret:
        :param instance_seg_ret:
        :return:
        """
        idx = np.where(binary_seg_ret == 255)
        lane_embedding_feats = instance_seg_ret[idx]
        # idx_scale = np.vstack((idx[0] / 256.0, idx[1] / 512.0)).transpose()
        # lane_embedding_feats = np.hstack((lane_embedding_feats, idx_scale))
        lane_coordinate = np.vstack((idx[1], idx[0])).transpose()

        assert lane_embedding_feats.shape[0] == lane_coordinate.shape[0]

        ret = {
            'lane_embedding_feats': lane_embedding_feats,
            'lane_coordinates': lane_coordinate
        }

        return ret

    def apply_lane_feats_cluster(self, binary_seg_result, instance_seg_result,eps=1, min_samples=1000):
        """

        :param binary_seg_result:
        :param instance_seg_result:
        :return:
        """
        # get embedding feats and coords
        get_lane_embedding_feats_result = self._get_lane_embedding_feats(
            binary_seg_ret=binary_seg_result,
            instance_seg_ret=instance_seg_result
        )

        # dbscan cluster
        dbscan_cluster_result = self._embedding_feats_dbscan_cluster(
            embedding_image_feats=get_lane_embedding_feats_result['lane_embedding_feats'],
            eps=eps, min_samples=min_samples
        )

        mask = np.zeros(shape=[binary_seg_result.shape[0], binary_seg_result.shape[1], 3], dtype=np.uint8)
        mobis_mask = 255*np.ones(shape=[binary_seg_result.shape[0], binary_seg_result.shape[1], 3], dtype=np.uint8)
        mobis_mask[...,2] = 0


        db_labels = dbscan_cluster_result['db_labels']
        unique_labels = dbscan_cluster_result['unique_labels']
        coord = get_lane_embedding_feats_result['lane_coordinates']

        if db_labels is None:
            return None, None

        lane_coords = []

        for index, label in enumerate(unique_labels.tolist()):
            if label == -1:
                continue
            idx = np.where(db_labels == label)
            pix_coord_idx = tuple((coord[idx][:, 1], coord[idx][:, 0]))
            mask[pix_coord_idx] = self._color_map[index]
            mobis_mask[pix_coord_idx] = self._mobis_map[index]
            lane_coords.append(coord[idx])

        return mask, lane_coords, mobis_mask


class LaneNetPostProcessor(object):
    """
    lanenet post process for lane generation
    """
    def __init__(self, ipm_remap_file_path='./data/tusimple_ipm_remap.yml'):
        """

        :param ipm_remap_file_path: ipm generate file path
        """
        # assert ops.exists(ipm_remap_file_path), '{:s} not exist'.format(ipm_remap_file_path)

        self._cluster = _LaneNetCluster()

        try:
            self._ipm_remap_file_path = ipm_remap_file_path
            remap_file_load_ret = self._load_remap_matrix()
            self._remap_to_ipm_x = remap_file_load_ret['remap_to_ipm_x']
            self._remap_to_ipm_y = remap_file_load_ret['remap_to_ipm_y']
        except:
            pass


        self._color_map = [np.array([255, 0, 0]),
                           np.array([0, 255, 0]),
                           np.array([0, 0, 255]),
                           np.array([125, 125, 0]),
                           np.array([0, 125, 125]),
                           np.array([125, 0, 125]),
                           np.array([50, 100, 50]),
                           np.array([100, 50, 100])]

    def _load_remap_matrix(self):
        """

        :return:
        """
        fs = cv2.FileStorage(self._ipm_remap_file_path, cv2.FILE_STORAGE_READ)

        remap_to_ipm_x = fs.getNode('remap_ipm_x').mat()
        remap_to_ipm_y = fs.getNode('remap_ipm_y').mat()

        ret = {
            'remap_to_ipm_x': remap_to_ipm_x,
            'remap_to_ipm_y': remap_to_ipm_y,
        }

        fs.release()

        return ret

    def postprocess(self, binary_seg_result, instance_seg_result=None,
                    min_area_threshold=100, source_image=None,
                    data_source='tusimple'):
        """

        :param binary_seg_result:
        :param instance_seg_result:
        :param min_area_threshold:
        :param source_image:
        :param data_source:
        :return:
        """
        # convert binary_seg_result
        binary_seg_result = np.array(binary_seg_result * 255, dtype=np.uint8)

        # apply image morphology operation to fill in the hold and reduce the small area
        morphological_ret = _morphological_process(binary_seg_result, kernel_size=5)

        show_source_image = np.zeros((1024,2048,3))

        show_source_image[512:,...] = np.transpose(source_image[0].cpu().numpy(),(1,2,0))

        # misc.imsave('/mfc/data/mobis/real/30_aa_seg_test/1438_20190418_173931_DL/1438_20190418_173931_00000000/seg_lane/temp.png',
        #             show_source_image + 3 * np.asarray(
        #                 np.stack((morphological_ret, morphological_ret, morphological_ret), axis=-1), dtype=np.uint8))
        # os.chmod('/mfc/data/mobis/real/30_aa_seg_test/1438_20190418_173931_DL/1438_20190418_173931_00000000/seg_lane/temp.png', 0o777)

        connect_components_analysis_ret = _connect_components_analysis(image=morphological_ret)

        labels = connect_components_analysis_ret[1]
        stats = connect_components_analysis_ret[2]
        for index, stat in enumerate(stats):
            if stat[4] <= min_area_threshold:
                idx = np.where(labels == index)
                morphological_ret[idx] = 0

        # misc.imsave(
        #     '/mfc/data/mobis/real/30_aa_seg_test/1438_20190418_173931_DL/1438_20190418_173931_00000000/seg_lane/temp1.png',
        #     np.transpose(source_image.cpu().numpy(), (1, 2, 0)) + 3 * np.asarray(
        #         np.stack((morphological_ret, morphological_ret, morphological_ret), axis=-1), dtype=np.uint8))
        # os.chmod(
        #     '/mfc/data/mobis/real/30_aa_seg_test/1438_20190418_173931_DL/1438_20190418_173931_00000000/seg_lane/temp1.png',
        #     0o777)


        # apply embedding features cluster
        mask_image, lane_coords, mobis_mask_image = self._cluster.apply_lane_feats_cluster(
            binary_seg_result=morphological_ret,
            instance_seg_result=instance_seg_result,eps=0.45, min_samples=100
        )

        lane_pts, lanepts_mask = getlane.getlane(mask_image,lane_coords)
        lanepts_mask  = np.stack((lanepts_mask, lanepts_mask, lanepts_mask), axis=-1)
        # misc.imsave(
        #     '/mfc/data/mobis/real/30_aa_seg_test/1438_20190418_173931_DL/1438_20190418_173931_00000000/seg_lane/temp2.png',
        #     mask_image)
        # os.chmod(
        #     '/mfc/data/mobis/real/30_aa_seg_test/1438_20190418_173931_DL/1438_20190418_173931_00000000/seg_lane/temp2.png',
        #     0o777)

        if mask_image is None:
            return {
                'mask_image': None,
                'fit_params': None,
                'source_image': None,
                'lane_pts':None,
            }
        # lane line fit
        fit_params = []
        src_lane_pts = []  # lane pts every single lane
        instance_imgs = []
        # for lane_index, coords in enumerate(lane_coords):
        #     tmp_mask = np.zeros(shape=(512, 512), dtype=np.uint8)
        #     tmp_mask[coords] = (lane_index+1)*100
        #     instance_imgs.append(tmp_mask)
        #     misc.imsave(
        #         '/mfc/data/mobis/real/30_aa_seg_test/1438_20190418_173931_DL/1438_20190418_173931_00000000/seg_lane/temp1_{}.png'.format(lane_index),
        #         np.transpose(source_image.cpu().numpy(), (1, 2, 0)) + 3 * np.asarray(
        #             np.stack((tmp_mask, tmp_mask, tmp_mask), axis=-1), dtype=np.uint8))
        #     os.chmod(
        #         '/mfc/data/mobis/real/30_aa_seg_test/1438_20190418_173931_DL/1438_20190418_173931_00000000/seg_lane/temp1_{}.png'.format(lane_index),
        #         0o777)
        #
        #
        # for lane_index, coords in enumerate(lane_coords):
        #     if data_source == 'tusimple':
        #         tmp_mask = np.zeros(shape=(512, 512), dtype=np.uint8)
        #         tmp_mask[tuple((np.int_(coords[:, 1] * 512 / 256), np.int_(coords[:, 0] * 512 / 512)))] = 255
        #     elif data_source == 'beec_ccd':
        #         tmp_mask = np.zeros(shape=(1350, 2448), dtype=np.uint8)
        #         tmp_mask[tuple((np.int_(coords[:, 1] * 1350 / 256), np.int_(coords[:, 0] * 2448 / 512)))] = 255
        #     else:
        #         raise ValueError('Wrong data source now only support tusimple and beec_ccd')
        #     tmp_ipm_mask = cv2.remap(
        #         tmp_mask,
        #         self._remap_to_ipm_x,
        #         self._remap_to_ipm_y,
        #         interpolation=cv2.INTER_NEAREST
        #     )
        #     nonzero_y = np.array(tmp_ipm_mask.nonzero()[0])
        #     nonzero_x = np.array(tmp_ipm_mask.nonzero()[1])
        #
        #     fit_param = np.polyfit(nonzero_y, nonzero_x, 2)
        #     fit_params.append(fit_param)
        #
        #     [ipm_image_height, ipm_image_width] = tmp_ipm_mask.shape
        #     plot_y = np.linspace(10, ipm_image_height, ipm_image_height - 10)
        #     fit_x = fit_param[0] * plot_y ** 2 + fit_param[1] * plot_y + fit_param[2]
        #     # fit_x = fit_param[0] * plot_y ** 3 + fit_param[1] * plot_y ** 2 + fit_param[2] * plot_y + fit_param[3]
        #
        #     lane_pts = []
        #     for index in range(0, plot_y.shape[0], 5):
        #         src_x = self._remap_to_ipm_x[
        #             int(plot_y[index]), int(np.clip(fit_x[index], 0, ipm_image_width - 1))]
        #         if src_x <= 0:
        #             continue
        #         src_y = self._remap_to_ipm_y[
        #             int(plot_y[index]), int(np.clip(fit_x[index], 0, ipm_image_width - 1))]
        #         src_y = src_y if src_y > 0 else 0
        #
        #         lane_pts.append([src_x, src_y])
        #
        #     src_lane_pts.append(lane_pts)
        #
        # # tusimple test data sample point along y axis every 10 pixels
        # source_image_width = source_image.shape[1]
        # for index, single_lane_pts in enumerate(src_lane_pts):
        #     single_lane_pt_x = np.array(single_lane_pts, dtype=np.float32)[:, 0]
        #     single_lane_pt_y = np.array(single_lane_pts, dtype=np.float32)[:, 1]
        #     if data_source == 'tusimple':
        #         start_plot_y = 240
        #         end_plot_y = 720
        #     elif data_source == 'beec_ccd':
        #         start_plot_y = 820
        #         end_plot_y = 1350
        #     else:
        #         raise ValueError('Wrong data source now only support tusimple and beec_ccd')
        #     step = int(math.floor((end_plot_y - start_plot_y) / 10))
        #     for plot_y in np.linspace(start_plot_y, end_plot_y, step):
        #         diff = single_lane_pt_y - plot_y
        #         fake_diff_bigger_than_zero = diff.copy()
        #         fake_diff_smaller_than_zero = diff.copy()
        #         fake_diff_bigger_than_zero[np.where(diff <= 0)] = float('inf')
        #         fake_diff_smaller_than_zero[np.where(diff > 0)] = float('-inf')
        #         idx_low = np.argmax(fake_diff_smaller_than_zero)
        #         idx_high = np.argmin(fake_diff_bigger_than_zero)
        #
        #         previous_src_pt_x = single_lane_pt_x[idx_low]
        #         previous_src_pt_y = single_lane_pt_y[idx_low]
        #         last_src_pt_x = single_lane_pt_x[idx_high]
        #         last_src_pt_y = single_lane_pt_y[idx_high]
        #
        #         if previous_src_pt_y < start_plot_y or last_src_pt_y < start_plot_y or \
        #                 fake_diff_smaller_than_zero[idx_low] == float('-inf') or \
        #                 fake_diff_bigger_than_zero[idx_high] == float('inf'):
        #             continue
        #
        #         interpolation_src_pt_x = (abs(previous_src_pt_y - plot_y) * previous_src_pt_x +
        #                                   abs(last_src_pt_y - plot_y) * last_src_pt_x) / \
        #                                  (abs(previous_src_pt_y - plot_y) + abs(last_src_pt_y - plot_y))
        #         interpolation_src_pt_y = (abs(previous_src_pt_y - plot_y) * previous_src_pt_y +
        #                                   abs(last_src_pt_y - plot_y) * last_src_pt_y) / \
        #                                  (abs(previous_src_pt_y - plot_y) + abs(last_src_pt_y - plot_y))
        #
        #         if interpolation_src_pt_x > source_image_width or interpolation_src_pt_x < 10:
        #             continue
        #
        #         lane_color = self._color_map[index].tolist()
        #         cv2.circle(source_image, (int(interpolation_src_pt_x),
        #                                   int(interpolation_src_pt_y)), 5, lane_color, -1)
        ret = {
            'mask_image': mask_image,
            'mobis_mask_image':mobis_mask_image,
            # 'fit_params': fit_params,
            'source_image': source_image,
            'lanepts_mask':lanepts_mask,
            'lane_pts':lane_pts
        }

        return ret
