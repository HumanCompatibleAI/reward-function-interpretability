import scipy.ndimage as nd


def zoom_to(img, width):
    n = width // img.shape[-2] + 1
    img = img.repeat(n, axis=-3).repeat(n, axis=-2)
    r = float(width) / img.shape[-2]
    zoom = [1] * (img.ndim - 3) + [r, r, 1]
    return nd.zoom(img, zoom, order=0, mode="nearest")
