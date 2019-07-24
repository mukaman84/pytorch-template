import  numpy as np
from BsplineModel.SaveImg import ImgSave

import scipy.misc as misc
import tensorflow as tf
try:
    import horovod.tensorflow as hvd
except:
    pass


from tensorflow.keras.models import Model
from tensorflow import keras
from tensorflow.keras.layers import Conv1D, AveragePooling1D, Flatten, Dense

from geomdl import BSpline, utilities,helpers
# from tensorflow import


#1)Define # of control points, Nofcpts
Nofcpts = 10

#2)Define the degree of the curve, D
DegreeOfCurve = 4

#3)Define the uniform knots, ti, for i<=n+d+1, from i=0
def GetUniformKnots():
    ti=[]
    for i in range(Nofcpts+DegreeOfCurve+2):
        if i<=DegreeOfCurve:
            ti.append(0)
        elif i<=Nofcpts:
            ti.append(((i-DegreeOfCurve)/(Nofcpts-DegreeOfCurve+1)))
        else:
            ti.append(1)

    ti=utilities.generate_knot_vector(DegreeOfCurve, Nofcpts+1)

    return ti


def getABsis(cpt,ti,t):
    if t>=ti[cpt] and t<ti[cpt+1]:
        return 1
    else:
        return 0


def BSplineBasis(ti,InsSegImg,samplepoints):
    BasisArray=np.zeros([len(samplepoints),Nofcpts+1],dtype=np.float32)
    # for cp in range(Nofcpts+1):
    Nij = np.zeros([len(samplepoints), Nofcpts + 2 + DegreeOfCurve, DegreeOfCurve + 1])
    for deg in range(DegreeOfCurve + 1):
        if deg == 0:
            for cpt in range(Nofcpts + DegreeOfCurve + 1):
                for pt, t in enumerate(samplepoints):
                    Nij[pt][cpt][deg] = getABsis(cpt, ti, t / samplepoints[-1])
        else:
            for cpt in range(Nofcpts + DegreeOfCurve + 2 - deg):
                for pt, t in enumerate(samplepoints):
                    try:
                        LEFT = (t / samplepoints[-1] - ti[cpt]) / (ti[cpt + deg] - ti[cpt])
                        RIGHT = (ti[cpt + deg + 1] - t / samplepoints[-1]) / (ti[cpt + deg + 1] - ti[cpt + 1])
                        Nij[pt][cpt][deg] = LEFT * Nij[pt][cpt][deg - 1] + RIGHT * Nij[pt][cpt + 1][deg - 1]
                    except ZeroDivisionError:
                        Nij[pt][cpt][deg] = 0

    return Nij

def find_span(tensor_knot,tensor_samplet):


    temp_sample = tf.expand_dims(tf.ones_like(tensor_samplet),0)

    extended_knots = tf.matmul(tf.expand_dims(tensor_knot,1),temp_sample)

    temp_sample = tf.expand_dims(tf.ones_like(tensor_knot), 1)


    extended_samples = tf.matmul(temp_sample,tf.expand_dims(tensor_samplet, 0))

    labels_true = tf.ones_like(extended_samples)
    labels_false = 100 * tf.ones_like(extended_samples)
    tensor_span = tf.math.argmin(tf.where(extended_knots>extended_samples,labels_true,labels_false),axis=0)
    tensor_span = tf.subtract(tensor_span,tf.ones_like(tensor_span))
    tensor_span = tf.concat([tf.expand_dims(tensor_span[:-1],1), tf.expand_dims(tf.expand_dims(tensor_span[-2],0),1)],0)
    tensor_span = tf.squeeze(tensor_span)
    return tensor_span

def find_basis(tensor_degree,tensor_knot,tensor_span,tensor_samplet):
    left = tf.ones([tf.constant((tensor_samplet.shape[0]),dtype=tf.int32),tensor_degree])
    right = tf.zeros([tf.cast(tensor_samplet.shape[0],dtype=tf.int32),tensor_degree])
    Nij = tf.zeros([tf.cast((tensor_samplet.shape[0]),dtype=tf.int32), tensor_degree])
    b= tf.expand_dims(tf.ones([tf.cast((tensor_samplet.shape[0]),dtype=tf.int32)]),1)
    Nij = tf.concat([b,Nij],axis=1)

    temp_knot = tf.expand_dims(tf.ones(tensor_degree), 0)
    tensor_knot_matrix = tf.matmul(tf.expand_dims(tensor_samplet,1),tf.cast(temp_knot,dtype=tf.float64))
    a=tf.cast(tf.expand_dims(tf.zeros_like(tensor_samplet),1),dtype=tf.float64)
    tensor_knot_matrix = tf.concat([a,tensor_knot_matrix],-1)
    tensor_span_index = tf.ones_like(tensor_span)


    # tensor_left_matrix_element = tf.add(tensor_span,tensor_span_index)

    # with tf.device('/gpu:0'):

    temp_left_matrix = tensor_span

    I=tf.constant(1)


    def cond_span(i, tensor_lefts, A_tensor_left, span_index):
        return tf.less(i, tensor_degree+1)

    def body_span(i, tensor_lefts, A_tensor_left, span_index):
        if i==1:
            tensor_lefts_init = tf.zeros_like(tensor_lefts)
            tensor_lefts = tf.concat([tensor_lefts_init, tf.expand_dims(A_tensor_left, 1)], -1)
        else:
            A_tensor_left = tf.subtract(A_tensor_left, span_index)
            tensor_lefts = tf.concat([tensor_lefts, tf.expand_dims(A_tensor_left, 1)], -1)
        return i+1, tensor_lefts, A_tensor_left, span_index

    _,tensor_span_matrix,_,_ = tf.while_loop(
        cond_span, body_span, [I, tf.expand_dims(tensor_span,1), tensor_span, tensor_span_index],shape_invariants=
                                        [I.get_shape(),
                                          tf.TensorShape([None, None]),tensor_span.get_shape(),tensor_span_index.get_shape()])



    tensor_left_matrix = tf.gather(tensor_knot,tensor_span_matrix)
    tensor_left_matrix = tf.subtract(tensor_knot_matrix,tensor_left_matrix)

    i0 = tf.constant(1)
    def cond_right(i,_tensor_lefts, A_tensor_left, span_index):
        return tf.less(i, tensor_degree+1)

    def body_right(i, tensor_lefts, A_tensor_left, span_index):
        if i==1:
            tensor_lefts_init = tf.zeros_like(tensor_lefts)
            tensor_lefts = tf.concat([tensor_lefts_init, tf.expand_dims(A_tensor_left, 1)], -1)
        else:
            A_tensor_left = tf.add(A_tensor_left, span_index)
            tensor_lefts = tf.concat([tensor_lefts,tf.expand_dims(A_tensor_left,1)],-1)
        return i+1, tensor_lefts, A_tensor_left, span_index

    # tensor_spans = tf.expand_dims(tensor_span,1)
    _,tensor_span_matrix,_,_ = tf.while_loop(
        cond_right, body_right, [i0, tf.expand_dims(tf.add(tensor_span,tensor_span_index),1), tf.add(tensor_span,tensor_span_index), tensor_span_index],shape_invariants=
                                        [i0.get_shape(),tf.TensorShape([None, None]),tensor_span.get_shape(),tensor_span_index.get_shape()]
    )



    tensor_right_matrix = tf.gather(tensor_knot,tensor_span_matrix)
    tensor_right_matrix = tf.subtract(tensor_right_matrix,tensor_knot_matrix)


    # temp = 0
    temp = tf.expand_dims(tf.zeros_like(tensor_samplet),1)

    Nij = tf.expand_dims(tf.ones_like(tensor_samplet),1)

    saved = tf.expand_dims(tf.zeros_like(tensor_samplet),1)
    Temp_Nij = tf.expand_dims(tf.zeros_like(tensor_samplet),1)

    i1 = tf.constant(1)
    r0 = tf.constant(0)

    def fi1(i, r, temp, Nij, Temp_Nij, saved, tensor_left_matrix, tensor_right_matrix):
        def fi1r0(i, r, temp, Nij, Temp_Nij, saved, tensor_left_matrix, tensor_right_matrix):
            Temp_Nij = tf.expand_dims(tf.gather(Nij, r, axis=1), -1)
            temp = tf.gather(tensor_left_matrix, i - r, axis=1) + tf.gather(tensor_right_matrix, r + 1, axis=1)
            temp = tf.divide(Temp_Nij, tf.expand_dims(temp, -1))
            Nij = saved + tf.multiply(tf.expand_dims(tf.gather(tensor_right_matrix, r + 1, axis=1), -1), temp)
            Temp_Nij = Nij
            saved = tf.multiply(tf.expand_dims(tf.gather(tensor_left_matrix, i - r, axis=1), -1), temp)
            return i, r + 1, temp, Nij, Temp_Nij, saved, tensor_left_matrix, tensor_right_matrix

        # print("fi1r0",callable(fi1r0))

        return tf.cond(tf.equal(r, 0), lambda: fi1r0(i, r, temp, Nij, Temp_Nij, saved, tensor_left_matrix, tensor_right_matrix), \
                       lambda: fi1rn0(i, r, temp, Nij, Temp_Nij, saved, tensor_left_matrix, tensor_right_matrix))
                       # lambda: fi1rn0(i, r, temp, Nij, Temp_Nij, saved, tensor_left_matrix, tensor_right_matrix))

        # return i, r + 1, temp, Nij, Temp_Nij, saved, tensor_left_matrix, tensor_right_matrix

    def fi1rn0(i, r, temp, Nij, Temp_Nij, saved, tensor_left_matrix, tensor_right_matrix):
        Nij = tf.concat([Temp_Nij, saved], 1)
        saved = tf.expand_dims(tf.zeros_like(tensor_samplet), 1)
        temp = tf.cast(temp,dtype=tf.float64)
        return i + 1 , 0, temp, Nij,Temp_Nij, saved, tensor_left_matrix, tensor_right_matrix

    def fin1(i, r, temp, Nij, Temp_Nij, saved, tensor_left_matrix, tensor_right_matrix):
        return tf.cond(tf.equal(r, 0), lambda : fin1r0(i, r, temp, Nij, Temp_Nij, saved, tensor_left_matrix, tensor_right_matrix), \
                       lambda: fin1rn0(i, r, temp, Nij, Temp_Nij, saved, tensor_left_matrix, tensor_right_matrix))

        # return i, r + 1, temp, Nij, Temp_Nij, saved, tensor_left_matrix, tensor_right_matrix

    def fin1r0(i, r, temp, Nij, Temp_Nij, saved, tensor_left_matrix, tensor_right_matrix):
        Temp_N = tf.expand_dims(tf.gather(Nij,r,axis=1), -1)
        temp = tf.gather(tensor_left_matrix, i-r, axis=1) + tf.gather(tensor_right_matrix, r+1, axis=1)
        temp = tf.divide(Temp_N,tf.expand_dims(temp,-1))
        temp_Nij = saved + tf.multiply(tf.expand_dims(tf.gather(tensor_right_matrix, r+1, axis=1),-1),temp)
        Nij = tf.concat([temp_Nij,Nij[...,1:]],1)
        saved = tf.multiply(tf.expand_dims(tf.gather(tensor_left_matrix, i-r, axis=1),-1),temp)
        return i, r+1, temp, Nij, Temp_Nij, saved, tensor_left_matrix, tensor_right_matrix

    def fin1rn0(i, r, temp, Nij, Temp_Nij, saved, tensor_left_matrix, tensor_right_matrix):
        return tf.cond(tf.equal(r, i), lambda : fin1rni(i, r, temp, Nij, Temp_Nij, saved, tensor_left_matrix, tensor_right_matrix), \
                       lambda: fin1ri(i, r, temp, Nij, Temp_Nij, saved, tensor_left_matrix, tensor_right_matrix))

    def fin1rni(i, r, temp, Nij, Temp_Nij, saved, tensor_left_matrix, tensor_right_matrix):
        Temp_N = tf.expand_dims(tf.gather(Nij, r, axis=1), -1)
        temp = tf.gather(tensor_left_matrix, i-r, axis=1) + tf.gather(tensor_right_matrix, r+1, axis=1)
        temp = tf.divide(Temp_N,tf.expand_dims(temp,-1))

        # temp_Nij = tf.concat([Nij,saved + tf.multiply(tf.expand_dims(tf.gather(tensor_right_matrix, r+1, axis=1),-1),temp)],1)
        temp_Nij = saved + tf.multiply(tf.expand_dims(tf.gather(tensor_right_matrix, r + 1, axis=1), -1), temp)

        def frim1(Nij, temp_Nij):
            Nij = tf.concat([Nij[..., :r], temp_Nij], 1)
            return Nij

        def frinm1(Nij, temp_Nij):
            a = Nij[..., :r]
            b = Nij[..., r + 1:]

            Nij = tf.concat([a, temp_Nij, b], 1)
            return Nij

        Nij = tf.cond(tf.equal(r, i-1), lambda : frim1(Nij, temp_Nij), \
                      lambda: frinm1(Nij, temp_Nij))

        saved = tf.multiply(tf.expand_dims(tf.gather(tensor_left_matrix, i-r, axis=1),-1),temp)
        return i, r+1, temp, Nij, Temp_Nij, saved, tensor_left_matrix, tensor_right_matrix

    def fin1ri(i, r, temp, Nij, Temp_Nij, saved, tensor_left_matrix, tensor_right_matrix):
        Nij = tf.concat([Nij, saved], 1)
        Temp_Nij = Nij
        temp = tf.cast(temp,dtype=tf.float64)
        saved = tf.expand_dims(tf.zeros_like(tensor_samplet), 1)
        return i + 1, 0, temp, Nij, Temp_Nij, saved, tensor_left_matrix, tensor_right_matrix


    def cond(i, r,_temp, Nij, Temp_Nij, saved, tensor_left_matrix, tensor_right_matrix):
        return tf.less(i, tensor_degree+1)

    def body(i, r, temp, Nij, Temp_Nij, saved, tensor_left_matrix, tensor_right_matrix):
        # check if i==1
        return tf.cond(tf.equal(i,1),lambda : fi1(i, r, temp, Nij, Temp_Nij, saved, tensor_left_matrix, tensor_right_matrix), \
                       lambda: fin1(i, r, temp, Nij, Temp_Nij, saved, tensor_left_matrix, tensor_right_matrix))

        # if i==1 and r == 0:
        #     Temp_Nij = tf.expand_dims(tf.gather(Nij,r,axis=1), -1)
        #     temp = tf.gather(tensor_left_matrix, i-r, axis=1) + tf.gather(tensor_right_matrix, r+1, axis=1)
        #     temp = tf.divide(Temp_Nij,tf.expand_dims(temp,-1))
        #     Nij = saved + tf.multiply(tf.expand_dims(tf.gather(tensor_right_matrix, r+1, axis=1),-1),temp)
        #     Temp_Nij = Nij
        #     saved = tf.multiply(tf.expand_dims(tf.gather(tensor_left_matrix, i-r, axis=1),-1),temp)
        #     return i, r+1, temp, Nij, Temp_Nij, saved, tensor_left_matrix, tensor_right_matrix
        #
        # elif i>1 and r == 0:
        #     Temp_N = tf.expand_dims(tf.gather(Nij,r,axis=1), -1)
        #     temp = tf.gather(tensor_left_matrix, i-r, axis=1) + tf.gather(tensor_right_matrix, r+1, axis=1)
        #     temp = tf.divide(Temp_N,tf.expand_dims(temp,-1))
        #     temp_Nij = saved + tf.multiply(tf.expand_dims(tf.gather(tensor_right_matrix, r+1, axis=1),-1),temp)
        #     Nij = tf.concat([temp_Nij,Nij[...,1:]],1)
        #     saved = tf.multiply(tf.expand_dims(tf.gather(tensor_left_matrix, i-r, axis=1),-1),temp)
        #     return i, r+1, temp, Nij, Temp_Nij, saved, tensor_left_matrix, tensor_right_matrix
        #
        # elif tf.cond(i>r):
        #         # tf.greater(i,r):
        #     # Temp_Nij = tf.concat([Temp_Nij,tf.expand_dims(tf.gather(Nij,r,axis=1),-1)],1)
        #     Temp_N = tf.expand_dims(tf.gather(Nij, r, axis=1), -1)
        #     temp = tf.gather(tensor_left_matrix, i-r, axis=1) + tf.gather(tensor_right_matrix, r+1, axis=1)
        #     temp = tf.divide(Temp_N,tf.expand_dims(temp,-1))
        #
        #     # temp_Nij = tf.concat([Nij,saved + tf.multiply(tf.expand_dims(tf.gather(tensor_right_matrix, r+1, axis=1),-1),temp)],1)
        #     temp_Nij = saved + tf.multiply(tf.expand_dims(tf.gather(tensor_right_matrix, r + 1, axis=1), -1), temp)
        #     if r == i-1:
        #         Nij = tf.concat([Nij[...,:r],temp_Nij],1)
        #     else:
        #         a = Nij[..., :r]
        #         b = Nij[..., r+1:]
        #
        #         Nij = tf.concat([a, temp_Nij,b], 1)
        #
        #     saved = tf.multiply(tf.expand_dims(tf.gather(tensor_left_matrix, i-r, axis=1),-1),temp)
        #     return i, r+1, temp, Nij, Temp_Nij, saved, tensor_left_matrix, tensor_right_matrix
        #
        # elif r == i:
        #     if i== 1:
        #         Nij = tf.concat([Temp_Nij,saved],1)
        #         saved = tf.expand_dims(tf.zeros_like(tensor_samplet), 1)
        #         return i + 1 , 0, temp, Nij,Temp_Nij, saved, tensor_left_matrix, tensor_right_matrix
        #     else:
        #         Nij = tf.concat([Nij, saved], 1)
        #         Temp_Nij = Nij
        #         saved = tf.expand_dims(tf.zeros_like(tensor_samplet), 1)
        #         return i + 1, 0, temp, Nij, Temp_Nij, saved, tensor_left_matrix, tensor_right_matrix


    _,_,_,N_basis,Temp_N_basis,_,_,_ = tf.while_loop(
        cond, body, [i1, r0, temp, Nij, Temp_Nij, saved, tensor_left_matrix, tensor_right_matrix],shape_invariants=\
                                        [i1.get_shape(),r0.get_shape(),tf.TensorShape([None, None]),tf.TensorShape([None, None]), \
                                         tf.TensorShape([None, None]),tf.TensorShape([None, None]),
                                         tensor_left_matrix.get_shape(),tensor_right_matrix.get_shape()])

    return N_basis








def find_curvepts(tensor_span,tensor_basis,tensor_degree,SamplingPoints,smaplingvalues):
    smaplingvalues = np.asarray(smaplingvalues)

    smaplingvalues = np.pad(smaplingvalues, ((0,2048), (0,0)), 'constant', constant_values=(0))
    smaplingvalues = smaplingvalues[:2048,:]

    smaplingvalues = tf.cast(tf.expand_dims(tf.transpose(tf.convert_to_tensor(smaplingvalues)), 0), dtype=tf.float64)


    def cond_cpt(i, _tensor_span, tensor_span_matrix):
        return tf.less(i, DegreeOfCurve + 1)

    def body_cpt(i, tensor_span, tensor_span_matrix):
        temp_span = tf.cast(tf.ones(tensor_span.shape[0]) * (4 - i), dtype=tf.int64)
        First_span = tf.gather(tensor_span, 0, axis=1)

        if i == 0:
            tensor_span_matrix = tf.expand_dims(tf.subtract(First_span, temp_span), axis=-1)
        else:
            tensor_span_matrix = tf.concat(
                [tensor_span_matrix, tf.expand_dims(tf.subtract(First_span, temp_span), axis=-1)], 1)

        return i + 1, tensor_span, tensor_span_matrix

        # tensor_spans = tf.expand_dims(tensor_span,1)

    _, _, tensor_span_matrix = tf.while_loop(
        cond_cpt, body_cpt, [0, tf.expand_dims(tensor_span, -1), tf.expand_dims(tensor_span, -1)])








    # return tensor_cpt_matrix

def GetBspline(InsSegImg):
    if InsSegImg.shape[0] > InsSegImg.shape[1]:
        Longaxis = InsSegImg.shape[0]
        SamplingPoints = []
        samplingdomain = []
        smaplingvalues = []
        smaplingResults = []
        for x_val in range(Longaxis):
            if len(np.where(InsSegImg[x_val,:,0])[0])>0:
                SampleValueSet = np.where(InsSegImg[x_val,:, 0])[0]
                SamplingPoints.append([x_val,SampleValueSet[int(round(len(SampleValueSet) / 2))]])
                samplingdomain.append(x_val)
                smaplingvalues.append(
                    [SampleValueSet[int(round(len(SampleValueSet) / 2))], np.float64(x_val) / np.float64(Longaxis - 1)])
                smaplingResults.append(SampleValueSet[int(round(len(SampleValueSet) / 2))])
                ti = GetUniformKnots()
    else:
        Longaxis = InsSegImg.shape[1]
        SamplingPoints = []
        samplingdomain = []
        smaplingvalues = []
        smaplingResults = []
        for y_val in range(Longaxis):
            if len(np.where(InsSegImg[:, y_val, 0])[0])>0:
                SampleValueSet = np.where(InsSegImg[:, y_val, 0])[0]
                SamplingPoints.append([SampleValueSet[int(round(len(SampleValueSet)/2))],y_val])
                samplingdomain.append(y_val)
                smaplingvalues.append([SampleValueSet[int(round(len(SampleValueSet)/2))],np.float64(y_val)/np.float64(Longaxis-1)])
                smaplingResults.append(SampleValueSet[int(round(len(SampleValueSet)/2))])
                ti = GetUniformKnots()
        # Nij = BSplineBasis(ti, InsSegImg, samplingdomain)
        # A= Nij[...,-1]

    tensor_degree = tf.constant(4)
    tensor_Ncpts = Nofcpts
    knot_vector = [0.0,0.0,0.0,0.0,0.0,1.0/7.0,2.0/7.0,3.0/7.0,4.0/7.0,5.0/7.0,6.0/7.0,1.0,1.0,1.0,1.0,1.0]
    tensor_knot = tf.constant(knot_vector,dtype=tf.float64)
    tensor_samplet = tf.divide(tf.constant(samplingdomain[:-1]), samplingdomain[-1])

    tensor_span = find_span(tensor_degree,tensor_knot,tensor_Ncpts,tensor_samplet)
    tensor_basis = find_basis(tensor_degree,tensor_knot,tensor_span,tensor_samplet)
    tensor_predict_sampled_matrix = find_curvepts(tensor_span,tensor_basis,tensor_degree,SamplingPoints,smaplingvalues)

    #The inputs of sampling variables
    tensor_sampling_inputs = tf.gather(tf.convert_to_tensor(smaplingvalues),0,axis=1)

    return smaplingvalues, tensor_sampling_inputs, tensor_basis, tensor_span


def frange(start, stop, step=1.0):
    """ Implementation of Python's ``range()`` function which works with floats.

    Reference to this implementation: https://stackoverflow.com/a/36091634

    :param start: start value
    :type start: float
    :param stop: end value
    :type stop: float
    :param step: increment
    :type step: float
    :return: float
    :rtype: generator
    """
    i = 0.0
    x = float(start)  # Prevent yielding integers.
    x0 = x
    epsilon = step / 2.0
    yield x  # always yield first value
    while x + epsilon < stop:
        i += 1.0
        x = x0 + i * step
        yield x
    if stop > x:
        yield stop

def GetBspline_from_sampled_points(bsplineMat,img_w,img_h,Legacy=False):
    # smaplingvalues = tf.convert_to_tensor(bsplineW)

    degree=4
    tensor_degree = tf.convert_to_tensor(degree)
    tensor_Ncpts = 10
    knot_vector = utilities.generate_knot_vector(degree, tensor_Ncpts)

    # smaplingvalues = tf.cast(image_resized, tf.float32) * (1 / 255) - 0.5

    # for bMat in bsplineMat:
    #     a = bMat[:, 0]/img_w
    #     b = bMat[:, 1] / img_h
    #     c = bMat[:, 2]
    #     d = np.stack([a,b,c], axis=-1)
    legacy=Legacy
    if legacy:
        tensor_sampling_inputs = [tf.convert_to_tensor(bmat[..., :2]) for bmat in bsplineMat]
    else:
        tensor_sampling_inputs_w = [tf.convert_to_tensor(bmat[..., 0]) for bmat in bsplineMat]
        tensor_sampling_inputs_h = [tf.convert_to_tensor(bmat[..., 1]) for bmat in bsplineMat]
        tensor_sampling_inputs = tf.stack([tensor_sampling_inputs_w,tensor_sampling_inputs_h])


    bsplineMat = [np.stack([bMat[:, 0]/img_w,bMat[:, 1]/img_h,bMat[:, 2]],axis=-1) \
                  for bMat in bsplineMat]

    bsplineMat_x = [np.stack([bMat[:, 0], bMat[:, 2]], axis=-1) \
                  for bMat in bsplineMat]

    bsplineMat_y = [np.stack([bMat[:, 1], bMat[:, 2]], axis=-1) \
                    for bMat in bsplineMat]


    tensor_knot = tf.convert_to_tensor(knot_vector,dtype=tf.float64)
    tensor_samplet = [tf.convert_to_tensor(bMat[:,-1]) for bMat in bsplineMat]
    # print("hvd local link tensor",str(hvd.rank()))
    tensor_span = [find_span(tensor_knot, samplet) for samplet in tensor_samplet]




    tensor_basis = [find_basis(tensor_degree, tensor_knot, tspan, tsample) \
                    for tspan, tsample in zip(tensor_span,tensor_samplet)]



    bsplineMat = [np.asarray(bmat) for bmat in bsplineMat]
    bsplineMat_x = [np.asarray(bmat) for bmat in bsplineMat_x]
    bsplineMat_y = [np.asarray(bmat) for bmat in bsplineMat_y]

    # bsplineMat = [tf.cast(tf.expand_dims(tf.transpose(tf.convert_to_tensor(bmat)), 0),
    #                      dtype=tf.float64) for bmat in bsplineMat]
    bsplineMat =[tf.cast(tf.transpose(tf.convert_to_tensor(bmat)),dtype = tf.float64) for bmat in bsplineMat]
    bsplineMat_x = [tf.cast(tf.transpose(tf.convert_to_tensor(bmat)), dtype=tf.float64) for bmat in bsplineMat_x]
    bsplineMat_y = [tf.cast(tf.transpose(tf.convert_to_tensor(bmat)), dtype=tf.float64) for bmat in bsplineMat_y]
    # smaplingvalues = tf.cast(image_resized, tf.float32) * (1 / 255) - 0.5

    return bsplineMat, bsplineMat_x,bsplineMat_y,tensor_sampling_inputs, tensor_basis, tensor_span





