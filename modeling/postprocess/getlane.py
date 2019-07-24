import numpy as np
import math

def getlane(mask,lane_coords):
    src_lane_pts = []

    tmp_ipm_mask = []
    fit_params = []
    tmp_mask = np.zeros(shape=(mask.shape[0], mask.shape[1]), dtype=np.uint8)

    for lane_index, coords in enumerate(lane_coords):
        tmp_mask = np.zeros(shape=(mask.shape[0], mask.shape[1]), dtype=np.uint8)
        tmp_mask[tuple((np.int_(coords[:, 1]), np.int_(coords[:, 0])))] = 255



        nonzero_y = np.array(tmp_mask.nonzero()[0])
        nonzero_x = np.array(tmp_mask.nonzero()[1])

        fit_param = np.polyfit(nonzero_y, nonzero_x, 2)
        fit_params.append(fit_param)

        [ipm_image_height, ipm_image_width] = tmp_mask.shape
        # plot_y = np.linspace(100, ipm_image_height, ipm_image_height - 10)
        # fit_x = fit_param[0] * plot_y ** 2 + fit_param[1] * plot_y + fit_param[2]
        # fit_x = fit_param[0] * plot_y ** 3 + fit_param[1] * plot_y ** 2 + fit_param[2] * plot_y + fit_param[3]

        lane_pts = []
        start_plot_y = np.min(nonzero_y)
        end_plot_y  =np.max(nonzero_y)
        step = int(math.floor((end_plot_y - start_plot_y) / 101))
        step_index = 0
        start_plot_x = np.min(nonzero_x)
        end_plot_x = np.max(nonzero_x)

        for step_index, plot_y in enumerate(np.linspace(start_plot_y, end_plot_y, 101)):
            src_x = fit_param[0] * plot_y ** 2 + fit_param[1] * plot_y + fit_param[2]
            # src_x = fit_param[0] * plot_y + fit_param[1]
            src_x = int(np.clip(src_x, start_plot_x, end_plot_x))


            lane_pts.append([int(np.round(src_x)), int(np.round(plot_y)),step_index/100])
            # step_index +=1

        src_lane_pts.append(lane_pts)


    tmp_mask = np.zeros(shape=(mask.shape[0], mask.shape[1]), dtype=np.uint8)
    src_lane_pts = np.asarray(src_lane_pts)
    for lane_index, coords in enumerate(src_lane_pts):
        tmp_mask[tuple((np.int_(coords[:, 1]), np.int_(coords[:, 0])))] = lane_index+1

    return src_lane_pts,tmp_mask
