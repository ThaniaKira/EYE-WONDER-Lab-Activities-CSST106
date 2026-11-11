"""Lab 03 — histogram equalization and contrast adjustment."""
import cv2
import numpy as np
from .common import base_result, output, num, hist_image


def process(img, form, files=None):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    equal = cv2.equalizeHist(gray)
    clip = num(form, 'clip', 2.0, 1, 8)
    tile = num(form, 'tile', 8, 2, 16, int)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile,tile)).apply(gray)
    contrast = num(form, 'contrast', 1.35, .5, 3)
    brightness = num(form, 'brightness', 12, -100, 100, int)
    adjusted = np.clip(img.astype(np.float32)*contrast + brightness,0,255).astype(np.uint8)
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
    color_equal = cv2.cvtColor(yuv, cv2.COLOR_YCrCb2BGR)
    result = base_result(img,'Histogram & Contrast',note='Global equalization stretches the overall distribution; CLAHE improves local contrast with a clip limit. Luminance-channel equalization preserves hue better than equalizing RGB channels separately.')
    result['outputs'] += [output('Global equalization',equal), output('CLAHE · adaptive',clahe),output('Contrast / brightness',adjusted), output('Color · luminance equalized',color_equal),output('Pixel-intensity histograms',hist_image([('Original',gray),('Global',equal),('CLAHE',clahe)]))]
    result['stats'] += [{'label':'Original std. dev.','value':f'{gray.std():.1f}'},{'label':'Equalized std. dev.','value':f'{equal.std():.1f}'},{'label':'CLAHE std. dev.','value':f'{clahe.std():.1f}'}]
    return result
