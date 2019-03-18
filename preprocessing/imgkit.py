
import numpy as np
from PIL import Image, ImageDraw

def draw_grids(image, grid_shape, color=(255,0,0), width=2):
    d = ImageDraw.Draw(image)
    gw, gh = grid_shape
    w, h = image.size[0], image.size[1]
    rw, rh = w/gw, h/gh
    for j in range(gh):
        d.line([1, j*rh, w-1, j*rh], fill=color, width=width)
    for i in range(gw):
        d.line([i*rw, 1, i*rw, h-1], fill=color, width=width)

def draw_boxes(image, boxes, color=(255,0,0), width=2):
    d = ImageDraw.Draw(image)
    for box in boxes:
        [x, y, w, h] = box[:4]
        d.rectangle([x,y,x+w,y+h], outline=color, width=width)

def crop(image, rect, boxes):
    image = image.crop(rect)
    boxes=np.array([[b[0]-rect[0], b[1]-rect[1], b[2], b[3]] for b in boxes])
    return image, boxes

def resize(image, size, boxes):
    sx = float(size[0])/image.size[0]
    sy = float(size[1])/image.size[1] 
    image = image.resize(size, Image.LINEAR)
    boxes = np.array([[b[0]*sx, b[1]*sy, b[2]*sx, b[3]*sy] for b in boxes])
    return image, boxes