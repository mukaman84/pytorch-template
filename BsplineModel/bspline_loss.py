import tensorflow as tf
from tensorflow.keras.losses import mean_squared_error
import numpy as np
import mvpuai
from geomdl import BSpline, utilities,helpers

def get_basis(b_spline_cpts):
    degree=4
    # spans = helpers.find_spans(degree, knot_vector, 6, knots, helpers.find_span_linear)
    x = b_spline_cpts[0::2]
    y = b_spline_cpts[1::2]


    b_spline_cpts = mvpuai.reshape_coords_to_tuples(b_spline_cpts)
    a = len(b_spline_cpts)
    knot_vector = utilities.generate_knot_vector(degree, len(b_spline_cpts))
    start = knot_vector[degree]
    stop = knot_vector[-(degree + 1)]


    knots = utilities.linspace(start, stop, 101, decimals=6)
    spans = helpers.find_spans(degree, knot_vector, len(b_spline_cpts), knots, helpers.find_span_linear)
    basis = helpers.basis_functions(degree, knot_vector, spans, knots)

    eval_points = []
    for idx in range(len(knots)):
        crvpt = [0.0 for _ in range(4)]
        for i in range(0, degree + 1):
            crvpt[:] = [crv_p + (basis[idx][i] * ctl_p) for crv_p, ctl_p in
                        zip(crvpt, b_spline_cpts[spans[idx] - degree + i])]

        eval_points.append(crvpt)

    return eval_points#


def BsplineLoss(tensor_cpts, tensor_sampling_inputs, tensor_basis, tensor_span, DegreeOfCurve):

    if tf.__version__ =="1.13.0-rc1":
        def cond_cpt(i, tensor_span, tensor_span_matrix):
            return tf.less(i, DegreeOfCurve +1)

        def body_cpt(i, tensor_span, tensor_span_matrix):
            temp_span = [tf.cast(tf.cast(tf.ones(tspan.shape[0]),dtype=tf.int32) * (tf.cast(DegreeOfCurve - i, dtype=tf.int32)), dtype=tf.int64) for tspan in tensor_span]
            First_span = [tf.gather(tspan, 0, axis=1) for tspan in tensor_span]

            if tf.equal(i,0):
                tensor_span_matrix = [tf.expand_dims(tf.subtract(fspan, tespan), axis=-1) for fspan, tespan in zip(First_span, temp_span)]
            else:
                tensor_span_matrix = [tf.concat(\
                    [tspan, tf.expand_dims(tf.subtract(fspan, tespan), axis=-1)], 1) \
                    for tspan, fspan, tespan in zip(tensor_span_matrix,First_span, temp_span)]

            return i + 1, tensor_span, tensor_span_matrix

            # tensor_spans = tf.expand_dims(tensor_span,1)

        x = tensor_cpts[0]
        y = tensor_cpts[1]

        # x = np.expand_dims(b_spline_cpts[0::2],0)
        # y = np.expand_dims(b_spline_cpts[1::2],0)

        tensor_span = [tf.expand_dims(tspan, -1) for tspan in tensor_span]
        i0=tf.constant(0)
        _, _, tensor_span_matrix = tf.while_loop(
            cond_cpt, body_cpt, [i0, tensor_span, tensor_span])

    else:
        def cond_cpt(i, t_span, t_span_matrix):
            return tf.less(i, DegreeOfCurve +1)

        def body_cpt(i, t_span, t_span_matrix):
            temp_span = tf.cast(tf.cast(tf.ones(t_span.shape[0]),dtype=tf.int32) * (tf.cast(DegreeOfCurve - i, dtype=tf.int32)), dtype=tf.int64)
            First_span = tf.gather(t_span, 0, axis=1)

            if tf.equal(i,0):
                t_span_matrix = tf.expand_dims(tf.subtract(First_span, temp_span), axis=-1)
            else:
                t_span_matrix = tf.concat([t_span_matrix, tf.expand_dims(tf.subtract(First_span, temp_span), axis=-1)], 1)

            return i + 1, t_span, t_span_matrix

            # tensor_spans = tf.expand_dims(tensor_span,1)

        x = tensor_cpts[0]
        y = tensor_cpts[1]

        # x = np.expand_dims(b_spline_cpts[0::2],0)
        # y = np.expand_dims(b_spline_cpts[1::2],0)

        tensor_span = [tf.expand_dims(tspan, -1) for tspan in tensor_span]

        i0=tf.constant(0)

        tensor_span_matrix=[]
        for t_span in tensor_span:
            # t_span_mat= tf.expand_dims(t_span, axis=-1)
            _, _, t_span_matrix = tf.while_loop(
                cond_cpt, body_cpt, [i0, t_span, t_span],shape_invariants=\
                    [i0.get_shape(),t_span.get_shape(),tf.TensorShape([None, None])])
            tensor_span_matrix.append(t_span_matrix)




    # for ind, tspan_matrix in enumerate(tensor_span_matrix):
    #     a = x[ind,:]
    #     b= tspan_matrix
    #     tf.gather(a,b, axis=-1)


    tensor_cpt_matrix_x = [tf.squeeze(tf.gather(x[ind,:], tspan_matrix, axis=-1)) for ind, tspan_matrix in enumerate(tensor_span_matrix)]
    tensor_cpt_matrix_x = [tf.multiply(basis, tf.cast(tspan_matrix,dtype=tf.float64)) \
                           for basis, tspan_matrix in zip(tensor_basis,tensor_cpt_matrix_x)]
    tensor_cpt_matrix_x = [tf.reduce_sum(tspan_matrix,axis=-1) for tspan_matrix in tensor_cpt_matrix_x]

    tensor_cpt_matrix_y = [tf.squeeze(tf.gather(y[ind,:], tspan_matrix, axis=-1)) for ind, tspan_matrix in enumerate(tensor_span_matrix)]
    tensor_cpt_matrix_y = [tf.multiply(basis, tf.cast(tspan_matrix,dtype=tf.float64)) for basis, tspan_matrix in zip(tensor_basis,tensor_cpt_matrix_y)]
    tensor_cpt_matrix_y = [tf.reduce_sum(tspan_matrix,axis=-1) for tspan_matrix in tensor_cpt_matrix_y]

    mse_x = [mean_squared_error(tf.cast(tsinputs[:, 0], dtype=tf.float64),
                       tf.cast(tcptmatrix_x, dtype=tf.float64)) for tsinputs,tcptmatrix_x in zip(tensor_sampling_inputs,tensor_cpt_matrix_x)]
    mse_y = [mean_squared_error(tf.cast(tsinputs[:, 1], dtype=tf.float64),
                                tf.cast(tcptmatrix_y, dtype=tf.float64)) for tsinputs, tcptmatrix_y in
             zip(tensor_sampling_inputs, tensor_cpt_matrix_y)]
    mse = [(_x + _y)/2 for _x,_y in zip(mse_x,mse_y)]
    return tf.reduce_mean(mse), tensor_cpt_matrix_x

def BsplineLoss_graph(tensor_basis, tensor_span_m, DegreeOfCurve):
    def BsplineLoss_fixed(y_true, y_pred):
        tensor_sampling_inputs = y_true
        tensor_cpts = y_pred

        # if tf.__version__ == "1.13.0-rc1":
        tensor_span = [tf.expand_dims(t_span_m, -1) for t_span_m in tensor_span_m]

        def cond_cpt(i, t_span, t_span_matrix):
            return tf.less(i, DegreeOfCurve + 1)

        def body_cpt(i, t_span, t_span_matrix):
            temp_span = tf.cast(
                tf.cast(tf.ones(t_span.shape[0]), dtype=tf.int32) * (tf.cast(DegreeOfCurve - i, dtype=tf.int32)),
                dtype=tf.int64)
            First_span = tf.gather(t_span, 0, axis=1)

            def fi0(First_span, temp_span):
                t_span_matrix = tf.expand_dims(tf.subtract(First_span, temp_span), axis=-1)
                return t_span_matrix

            def fin0(t_span_matrix, First_span, temp_span):
                t_span_matrix = tf.concat(
                    [t_span_matrix, tf.expand_dims(tf.subtract(First_span, temp_span), axis=-1)], 1)
                return t_span_matrix

            # if tf.equal(i,0):
            #     t_span_matrix = tf.expand_dims(tf.subtract(First_span, temp_span), axis=-1)
            # else:
            #     t_span_matrix = tf.concat(
            #         [t_span_matrix, tf.expand_dims(tf.subtract(First_span, temp_span), axis=-1)], 1)

            t_span_matrix = tf.cond(tf.equal(i, 0), lambda: fi0(First_span, temp_span), \
                    lambda: fin0(t_span_matrix, First_span, temp_span))

            return i + 1, t_span, t_span_matrix

        x = tensor_cpts[0]
        y = tensor_cpts[1]

        i0 = tf.constant(0)

        tensor_span_matrix = []
        for t_span in tensor_span:
            # t_span_mat= tf.expand_dims(t_span, axis=-1)
            _, _, t_span_matrix = tf.while_loop(
                cond_cpt, body_cpt, [i0, t_span, t_span], shape_invariants= \
                    [i0.get_shape(), t_span.get_shape(), tf.TensorShape([None, None])])
            tensor_span_matrix.append(t_span_matrix)

        tensor_cpt_matrix_x = [tf.squeeze(tf.gather(x[ind, :], tspan_matrix, axis=-1)) for ind, tspan_matrix in enumerate(tensor_span_matrix)]
        tensor_cpt_matrix_x = [tf.multiply(basis, tf.cast(tspan_matrix, dtype=tf.float64)) for basis, tspan_matrix in zip(tensor_basis, tensor_cpt_matrix_x)]
        tensor_cpt_matrix_x = [tf.reduce_sum((tspan_matrix), axis=-1) for tspan_matrix in tensor_cpt_matrix_x]

        tensor_cpt_matrix_y = [tf.squeeze(tf.gather(y[ind, :], tspan_matrix, axis=-1)) for ind, tspan_matrix in enumerate(tensor_span_matrix)]
        tensor_cpt_matrix_y = [tf.multiply(basis, tf.cast(tspan_matrix, dtype=tf.float64)) for basis, tspan_matrix in zip(tensor_basis, tensor_cpt_matrix_y)]
        tensor_cpt_matrix_y = [tf.reduce_sum((tspan_matrix), axis=-1) for tspan_matrix in tensor_cpt_matrix_y]

        tsinputs_x = tensor_sampling_inputs[0]
            # tf.map_fn(lambda x:tf.cast(x[...,0], dtype=tf.float64),tensor_sampling_inputs)
            # [tf.cast(tsinputs[:, 0], dtype=tf.float64) for tsinputs in tensor_sampling_inputs]
        tsinputs_y = tensor_sampling_inputs[1]
            # tf.map_fn(lambda x:tf.cast(x[...,1], dtype=tf.float64),tensor_sampling_inputs)
            # [tf.cast(tsinputs[:, 1], dtype=tf.float64) for tsinputs in tensor_sampling_inputs]

        tensor_cpt_matrix_x = [tf.cast(tcptmatrix_x, dtype=tf.float32) for tcptmatrix_x in tensor_cpt_matrix_x]
        tensor_cpt_matrix_y = [tf.cast(tcptmatrix_y, dtype=tf.float32) for tcptmatrix_y in tensor_cpt_matrix_y]


        # mse_x = tf.map_fn(lambda x: mean_squared_error(x[0], x[1]), (tsinputs_x, tensor_cpt_matrix_x))
        # mse_y = tf.map_fn(lambda x: mean_squared_error(x[0], x[1]), (tsinputs_y, tensor_cpt_matrix_y))

        mse_x = [mean_squared_error(tsinputs_x[ind],tcptmatrix_x) for ind, tcptmatrix_x in enumerate(tensor_cpt_matrix_x)]
        mse_y = [mean_squared_error(tsinputs_y[ind],tcptmatrix_y) for ind, tcptmatrix_y in enumerate(tensor_cpt_matrix_y)]
            # [mean_squared_error(tsinputs,tcptmatrix_y) for tsinputs, tcptmatrix_y in \
            #      zip(tsinputs_y, tensor_cpt_matrix_y)]
        mse = [(_x + _y) / 2 for _x, _y in zip(mse_x, mse_y)]
        # tf.map_fn(lambda x: x[0] + x[1], (mse_x, mse_y))
            # [(_x + _y) / 2 for _x, _y in zip(mse_x, mse_y)]
        return tf.reduce_mean(mse)



        # mse_x = [mean_squared_error(tf.cast(tsinputs[:, 0], dtype=tf.float64),
        #                             tf.cast(tcptmatrix_x, dtype=tf.float64)) for tsinputs, tcptmatrix_x in
        #          zip(tensor_sampling_inputs, tensor_cpt_matrix_x)]


        # sampling_input_x = tf.squeeze(tensor_sampling_inputs[...,0])
        # sampling_input_y = tf.squeeze(tensor_sampling_inputs[..., 1])
        #
        # mse_x = mean_squared_error(sampling_input_x,tensor_cpt_matrix_x)
        #
        # mse_y = mean_squared_error(sampling_input_y, tensor_cpt_matrix_y)


        # mse_y = [mean_squared_error(tf.cast(tsinputs[:, 1], dtype=tf.float64),
        #                             tf.cast(tcptmatrix_y, dtype=tf.float64)) for tsinputs, tcptmatrix_y in
        #          zip(tensor_sampling_inputs, tensor_cpt_matrix_y)]
        # mse = [(_x + _y) / 2 for _x, _y in zip(mse_x, mse_y)]
        # mse = mse_x + mse_y
        # return mse/2
        # else:
        #     tensor_span = tensor_span_m
        #     def cond_cpt(i, t_span, t_span_matrix):
        #         return tf.less(i, DegreeOfCurve + 1)
        #
        #     def body_cpt(i, t_span, t_span_matrix):
        #         temp_span = tf.cast(
        #             tf.cast(tf.ones(t_span.shape[0]), dtype=tf.int32) * (tf.cast(DegreeOfCurve - i, dtype=tf.int32)),
        #             dtype=tf.int64)
        #         First_span = tf.gather(t_span, 0, axis=1)
        #
        #         def fi0(First_span, temp_span):
        #             t_span_matrix = tf.expand_dims(tf.subtract(First_span, temp_span), axis=-1)
        #             return t_span_matrix
        #
        #         def fin0(t_span_matrix, First_span, temp_span):
        #             t_span_matrix = tf.concat(
        #                 [t_span_matrix, tf.expand_dims(tf.subtract(First_span, temp_span), axis=-1)], 1)
        #             return t_span_matrix
        #
        #         # if tf.equal(i,0):
        #         #     t_span_matrix = tf.expand_dims(tf.subtract(First_span, temp_span), axis=-1)
        #         # else:
        #         #     t_span_matrix = tf.concat(
        #         #         [t_span_matrix, tf.expand_dims(tf.subtract(First_span, temp_span), axis=-1)], 1)
        #
        #         t_span_matrix = tf.cond(tf.equal(i, 0), lambda: fi0(First_span, temp_span), \
        #                 lambda: fin0(t_span_matrix, First_span, temp_span))
        #         return i + 1, t_span, t_span_matrix
        #
        #
        #         # tensor_spans = tf.expand_dims(tensor_span,1)
        #
        #     x = tensor_cpts[0]
        #     y = tensor_cpts[1]
        #
        #     # x = np.expand_dims(b_spline_cpts[0::2],0)
        #     # y = np.expand_dims(b_spline_cpts[1::2],0)
        #
        #     tensor_span = [tf.expand_dims(tspan, -1) for tspan in tensor_span]
        #
        #     i0 = tf.constant(0)
        #
        #     tensor_span_matrix = []
        #     for t_span in tensor_span:
        #         # t_span_mat= tf.expand_dims(t_span, axis=-1)
        #         _, _, t_span_matrix = tf.while_loop(
        #             cond_cpt, body_cpt, [i0, t_span, t_span], shape_invariants= \
        #                 [i0.get_shape(), t_span.get_shape(), tf.TensorShape([None, None])])
        #         tensor_span_matrix.append(t_span_matrix)
        #
        #     tensor_cpt_matrix_x = [tf.squeeze(tf.gather(x[ind, :], tspan_matrix, axis=-1)) for ind, tspan_matrix in
        #                            enumerate(tensor_span_matrix)]
        #     tensor_cpt_matrix_x = [tf.multiply(basis, tf.cast(tspan_matrix, dtype=tf.float64)) \
        #                            for basis, tspan_matrix in zip(tensor_basis, tensor_cpt_matrix_x)]
        #     tensor_cpt_matrix_x = [tf.reduce_sum(tspan_matrix, axis=-1) for tspan_matrix in tensor_cpt_matrix_x]
        #
        #     tensor_cpt_matrix_y = [tf.squeeze(tf.gather(y[ind, :], tspan_matrix, axis=-1)) for ind, tspan_matrix in
        #                            enumerate(tensor_span_matrix)]
        #     tensor_cpt_matrix_y = [tf.multiply(basis, tf.cast(tspan_matrix, dtype=tf.float64)) for basis, tspan_matrix
        #                            in zip(tensor_basis, tensor_cpt_matrix_y)]
        #     tensor_cpt_matrix_y = [tf.reduce_sum(tspan_matrix, axis=-1) for tspan_matrix in tensor_cpt_matrix_y]
        #
        #     mse_x = [mean_squared_error(tf.cast(tsinputs[:, 0], dtype=tf.float64),
        #                                 tf.cast(tcptmatrix_x, dtype=tf.float64)) for tsinputs, tcptmatrix_x in
        #              zip(tensor_sampling_inputs, tensor_cpt_matrix_x)]
        #     mse_y = [mean_squared_error(tf.cast(tsinputs[:, 1], dtype=tf.float64),
        #                                 tf.cast(tcptmatrix_y, dtype=tf.float64)) for tsinputs, tcptmatrix_y in
        #              zip(tensor_sampling_inputs, tensor_cpt_matrix_y)]
        #     mse = [(_x + _y) / 2 for _x, _y in zip(mse_x, mse_y)]
        #     return tf.reduce_mean(mse)
        # return tf.reduce_mean(mse)
    return BsplineLoss_fixed












    # return mean_squared_error(tf.cast(tensor_sampling_inputs[:,0],dtype=tf.float64),tf.cast(tensor_cpt_matrix_x,dtype=tf.float64))+ \
    #        mean_squared_error(tf.cast(tensor_sampling_inputs[:,1], dtype=tf.float64),
    #                           tf.cast(tensor_cpt_matrix_y, dtype=tf.float64))





