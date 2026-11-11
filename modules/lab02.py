"""Lab 02 — Sobel and Canny edge maps."""
import cv2
from .common import base_result, output, num


def process(img, form, files=None):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    low = num(form, 'canny_low', 80, 0, 254, int)
    high = num(form, 'canny_high', 190, low+1, 255, int)
    sx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(sx, sy)
    sobel = cv2.convertScaleAbs(magnitude)
    canny = cv2.Canny(blur, low, high)
    result = base_result(img, 'Edge Detection', note='Sobel displays gradient strength; Canny returns a thinner, thresholded edge map. Adjust thresholds to compare noise sensitivity.')
    result['outputs'] += [output('Gaussian blur · 3×3',blur), output('Sobel · gradient magnitude',sobel), output(f'Canny · {low}/{high}',canny)]
    result['stats'] += [{'label':'Canny edge pixels', 'value': f'{cv2.countNonZero(canny):,}'}, {'label':'Edge coverage','value': f'{100*cv2.countNonZero(canny)/canny.size:.1f}%'}]
    return result
