import scipy.misc as misc
import os




def ImgSave(aSegFile,loadedSegImg,LaneId=''):
    aSegFile = aSegFile.split('.')[0]
    if LaneId=='':
        SegName = "/mfc/user/1623600/.temp/{:s}.png".format(os.path.basename(aSegFile))
    else:
        SegName = "/mfc/user/1623600/.temp/{:s}_{:s}.png".format(os.path.basename(aSegFile), str(LaneId))

    try:
        misc.imsave(SegName, loadedSegImg)
        os.chmod(SegName, 0o777)
    except :
        print("stop cos this is an error")


