import numpy as np


def hsv_to_rgb(h, s, v):
    if s == 0.0:
        return v, v, v
    i = int(h*6.0) # XXX assume int() truncates!
    f = (h*6.0) - i
    p = v*(1.0 - s)
    q = v*(1.0 - s*f)
    t = v*(1.0 - s*(1.0-f))
    i = i%6
    if i == 0:
        return v, t, p
    if i == 1:
        return q, v, p
    if i == 2:
        return p, v, t
    if i == 3:
        return p, q, v
    if i == 4:
        return t, p, v
    if i == 5:
        return v, p, q
    # Cannot get here


def to_color(category):
    """Map each category color a good distance away from each other on the HSV color space."""
    v = (category - 1) * (137.5 / 360)
    return hsv_to_rgb(v, 1, 1)

def add_color(img, num_classes=12):
    h, w = img.shape
    img_color = np.zeros((h, w, 3))
    for i in range(1, 12):
        img_color[img == i] = to_color(i+1)
    # img_color[img == num_classes] = (1.0, 1.0, 1.0)
    return img_color


