from modeling.extension import unet

def build_extension(ext, out_channels,kernel_size,padding,n_resblocks):
    if ext == 'unet':
        return unet.UnetDecoder(out_channels,kernel_size,padding,n_resblocks)
    else:
        raise NotImplementedError
