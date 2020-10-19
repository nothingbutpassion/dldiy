
import numpy as np
from PIL import Image, ImageDraw

def draw_boxes(image, boxes, color=(255,0,0), width=2):
    d = ImageDraw.Draw(image)
    for box in boxes:
        [x, y, w, h] = box[:4]
        d.rectangle([x,y,x+w,y+h], outline=color, width=width)
        d.point([(x+w/2-0.5,y+h/2-0.5), (x+w/2+0.5,y+h/2-0.5), (x+w/2-0.5,y+h/2+0.5), (x+w/2+0.5,y+h/2+0.5)], fill=color)

def crop(image, rect, boxes=None):
    image = image.crop(rect)
    if boxes is None:
        return image
    boxes=np.array([[b[0]-rect[0], b[1]-rect[1], b[2], b[3]] for b in boxes])
    return image, boxes

def resize(image, size, boxes=None):
    sx = float(size[0])/image.size[0]
    sy = float(size[1])/image.size[1] 
    image = image.resize(size, Image.LINEAR)
    if boxes is None:
        return image
    boxes = np.array([[b[0]*sx, b[1]*sy, b[2]*sx, b[3]*sy] for b in boxes])
    return image, boxes

# flag can be one of the following:
# Image.FLIP_LEFT_RIGHT
# Image.FLIP_TOP_BOTTOM
def flip(image, boxes=None, flag=Image.FLIP_LEFT_RIGHT):
    image = image.transpose(flag)
    if boxes is None:
        return image
    w, h = image.size
    if flag == Image.FLIP_LEFT_RIGHT:
        boxes = np.array([[w-1-b[0]-b[2], b[1], b[2], b[3]] for b in boxes])
    else:
        boxes = np.array([[b[0], h-1-b[1]-b[3], b[2], b[3]] for b in boxes])
    return image, boxes