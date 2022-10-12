from lucent.optvis.param.spatial import pixel_image, fft_image
from lucent.optvis.param.color import to_valid_rgb

def image(w, h=None, sd=None, batch=None, decorrelate=True,
          fft=True, channels=None):
    h = h or w
    batch = batch or 1
    ch = channels or 3
    shape = [batch, h, w, ch]
    param_f = fft_image if fft else pixel_image
    params, image_f = param_f(shape, sd=sd)
    if channels:
        output = to_valid_rgb(image_f, decorrelate=False)
    else:
        output = to_valid_rgb(image_f, decorrelate=decorrelate)
    return params, output