from BsplineModel.SaveImg import ImgSave
import numpy as np
from scipy.spatial import distance
from BsplineModel.GetBspline import GetBspline

def GetBoundOfAxis(MaxorMin,LenghLongAxis):
    if np.min(MaxorMin) < (LenghLongAxis/2):
        return np.max(MaxorMin)
    else:
        return np.min(MaxorMin)

def GetReverseBoundOfAxis(MaxorMin,LenghLongAxis):
    if np.min(MaxorMin) > (LenghLongAxis/2):
        return np.max(MaxorMin)
    else:
        return np.min(MaxorMin)

def GetBoundedBox(ShortShpe,LongShpe, minInsSegImg,maxInsSegImg,btmShortInsSegImg,topShortInsSegImg,shape_d):
    ResizeBoxSmax = ShortShpe - 1
    ResizeBoxSmin = 0


    #Check that the coordinates of 255 value of short axis is left or right
    if len(np.where(maxInsSegImg == 255)[0]) > 0:
        MaxorMin_one = GetBoundOfAxis(np.where(maxInsSegImg == 255), LongShpe)
        if len(np.where(minInsSegImg == 255)[0]) > 0:  # First, check that the direction of coordinates of 0 value of short axis
            MaxorMin_two = GetBoundOfAxis(np.where(minInsSegImg == 255), LongShpe)
        else:
            if MaxorMin_one > (LongShpe/2):
                MaxorMin_two = 0
            else:
                MaxorMin_two = LongShpe-1


    else:#This works when we don't know the direction of 255 value
        if len(np.where(minInsSegImg == 255)[0]) > 0:#First, check that the direction of coordinates of 0 value of short axis
            MaxorMin_two = GetBoundOfAxis(np.where(minInsSegImg == 255), LongShpe)
            if MaxorMin_two > (LongShpe/2):
                MaxorMin_one = 0
            else:
                MaxorMin_one = LongShpe-1
        else:
            MaxorMin_one = -1
            MaxorMin_two = -1

    if MaxorMin_one > MaxorMin_two:
        ResizeBoxLmaxShort = ShortShpe - 1
        ResizeBoxLminShort = 0
        ResizeBoxLmax = MaxorMin_one
        ResizeBoxLmin = MaxorMin_two

    else:
        ResizeBoxLmaxShort = 0
        ResizeBoxLminShort = ShortShpe - 1
        ResizeBoxLmax = MaxorMin_two
        ResizeBoxLmin = MaxorMin_one



    if len(np.where(btmShortInsSegImg == 255)[0]) > 0 and len(np.where(topShortInsSegImg == 255)[0]) > 0:
        btmValueShortAxis = GetReverseBoundOfAxis(np.where(btmShortInsSegImg == 255), ShortShpe)
        topValueShortAxis = GetReverseBoundOfAxis(np.where(topShortInsSegImg == 255), ShortShpe)

        if not shape_d == 0:  # for width
            btmShortAxis = [btmValueShortAxis,0]
            topShortAxis = [topValueShortAxis,LongShpe - 1]
            btmLongAxis = [ResizeBoxLminShort,ResizeBoxLmin]
            topLongAxis = [ResizeBoxLmaxShort,ResizeBoxLmax]
        else:
            btmShortAxis = [0, btmValueShortAxis]
            topShortAxis = [LongShpe - 1, topValueShortAxis]
            btmLongAxis = [ResizeBoxLmin, ResizeBoxLminShort]
            topLongAxis = [ResizeBoxLmax, ResizeBoxLmaxShort]
    else:
        return ResizeBoxSmin, ResizeBoxSmax, ResizeBoxLmin, ResizeBoxLmax




    distance1 = np.abs(np.subtract(btmShortAxis,btmLongAxis))
    distance2 = np.abs(np.subtract(btmShortAxis,topLongAxis))
    distance3 = np.abs(np.subtract(topShortAxis,btmLongAxis))
    distance4 = np.abs(np.subtract(topShortAxis,topLongAxis))



    distance1RatioW = distance1[0] / (ShortShpe -1)
    distance1RatioH = distance1[1] / (LongShpe -1)
    distance4RatioW = distance4[0] / (ShortShpe - 1)
    distance4RatioH = distance4[1] / (LongShpe - 1)
    if (distance1RatioW > 0.05 and distance1RatioH > 0.05 and distance1[0] > 10 and distance1[1] > 10) or (distance4RatioW > 0.1 and distance4RatioH > 0.1 and distance4[0] > 10 and distance4[1] > 10):
        ResizeBoxLmin = -1
        ResizeBoxLmax = -1
        return ResizeBoxSmin, ResizeBoxSmax, ResizeBoxLmin, ResizeBoxLmax
    else:
        return ResizeBoxSmin, ResizeBoxSmax, ResizeBoxLmin, ResizeBoxLmax

def CustomCropMask(aSegFile, InsSegImg, LaneId):
    if InsSegImg.shape[0] > InsSegImg.shape[1]:
        ResizeBoxXmin, ResizeBoxXmax, ResizeBoxYmin, ResizeBoxYmax = \
            GetBoundedBox(InsSegImg.shape[1], InsSegImg.shape[0], InsSegImg[:, 0, 0], InsSegImg[:, -1, 0],InsSegImg[0, :, 0],InsSegImg[-1, :, 0],0)
    else:
        #Set default values for short axis
        ResizeBoxYmin, ResizeBoxYmax, ResizeBoxXmin, ResizeBoxXmax=\
            GetBoundedBox(InsSegImg.shape[0],InsSegImg.shape[1],InsSegImg[0, :, 0],InsSegImg[-1, :, 0], InsSegImg[:, 0, 0], InsSegImg[:, -1, 0],1)

    if ResizeBoxXmin==ResizeBoxXmax==-1:
        InsSegImg = InsSegImg[ResizeBoxYmin:ResizeBoxYmax, :, :]
    elif ResizeBoxYmin==ResizeBoxYmax==-1:
        InsSegImg = InsSegImg[:, ResizeBoxXmin:ResizeBoxXmax, :]
    else:
        InsSegImg = InsSegImg[ResizeBoxYmin:ResizeBoxYmax, ResizeBoxXmin:ResizeBoxXmax, :]

    #Here, the function to get b-spline
    GetBspline(InsSegImg)

    if len(InsSegImg)> 0:
        ImgSave(aSegFile, InsSegImg, str(LaneId)+'_Recrop')






def CropInsLane(aSegFile, InsSegImg, LaneId=''):
    a = np.where(InsSegImg[..., 0] == 255)
    boxXmin = np.min(np.where(InsSegImg[..., 0] == 255)[0])
    boxYmin = np.min(np.where(InsSegImg[..., 0] == 255)[1])
    boxXmax = np.max(np.where(InsSegImg[..., 0] == 255)[0])
    boxYmax = np.max(np.where(InsSegImg[..., 0] == 255)[1])

    InsSegImg = InsSegImg[boxXmin:boxXmax,boxYmin:boxYmax,:]
    ImgSave(aSegFile, InsSegImg, LaneId)


    CustomCropMask(aSegFile, InsSegImg, LaneId)


