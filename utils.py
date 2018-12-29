import numpy as np

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    # NOTES
    # orignal input_data.shape: (N, C, H, W)
    # after pad: (N, C, H + 2*pad, W + 2*pad)
    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    # NOTES 
    # orignal col.shape: (N, C, filter_h, filter_w, out_h, out_w)
    # after transpose:   (N, out_h, out_w, C, filter_h, filter_w) 
    # after reshape:     (N*out_h*out_w, C*filter_h*filter_w)
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col

def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    # NOTES
    # orignal col.shape (N*out_h*out_w, C*filter_h*filter_w)
    # after reshape:    (N, out_h, out_w, C, filter_h, filter_w)
    # after transpose:  (N, C, filter_h, filter_W, out_h, out_w)
    col = col.reshape((N, out_h, out_w, C, filter_h, filter_w)).transpose(0, 3, 4, 5, 1, 2)
    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] = col[:, :, y, x, :, :]
    return img[:, :, pad:pad+H + pad, pad:pad+W]





