"""Lab 01 — image representation & manipulation."""
import cv2
from .common import base_result, num, output


def process(img, form, files=None):
    h, w = img.shape[:2]
    result = base_result(img, 'Image Manipulation', note='OpenCV stores color pixels in BGR order. The browser shows images converted to standard encoded color format.')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    result['outputs'].append(output('Grayscale', gray, 'Single-channel luminance representation'))

    zoom = num(form, 'scale', 60, 20, 200, int)
    resized = cv2.resize(img, (max(1,int(w*zoom/100)), max(1,int(h*zoom/100))), interpolation=cv2.INTER_AREA if zoom < 100 else cv2.INTER_CUBIC)
    result['outputs'].append(output(f'Resized · {zoom}%', resized))

    rotate = int(num(form, 'rotation', 90, 0, 270, int))
    rotated = {0:img,90:cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),180:cv2.rotate(img, cv2.ROTATE_180),270:cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)}.get(rotate, img)
    result['outputs'].append(output(f'Rotated · {rotate}°', rotated))

    crop_pct = num(form, 'crop', 65, 20, 95, int) / 100
    crop_w, crop_h = max(1, int(w*crop_pct)), max(1, int(h*crop_pct))
    left, top = (w-crop_w)//2, (h-crop_h)//2
    result['outputs'].append(output('Center crop · ROI', img[top:top+crop_h, left:left+crop_w]))

    x = num(form, 'pixel_x', w//2, 0, w-1, int)
    y = num(form, 'pixel_y', h//2, 0, h-1, int)
    edited = img.copy()
    b, g, r = (int(c) for c in img[y,x])
    hex_color = form.get('pixel_color', '#f85acb')
    if len(hex_color) != 7 or not hex_color.startswith('#'):
        hex_color = '#f85acb'
    try:
        rr, gg, bb = (int(hex_color[k:k+2],16) for k in (1,3,5))
    except ValueError:
        rr, gg, bb = 248, 90, 203
    edited[y, x] = (bb, gg, rr)
    result['outputs'].append(output('Pixel color edited', edited, f'Pixel ({x}, {y}): RGB ({r}, {g}, {b}) → ({rr}, {gg}, {bb})'))
    # Magnify a small region using nearest-neighbor so the exact edited pixel is visible.
    radius = 7
    lx, rx = max(0, x-radius), min(w, x+radius+1)
    ty, by = max(0, y-radius), min(h, y+radius+1)
    zoomed = cv2.resize(edited[ty:by, lx:rx], (300, 300), interpolation=cv2.INTER_NEAREST)
    centerx, centery = (x-lx+.5)*300/(rx-lx), (y-ty+.5)*300/(by-ty)
    cv2.rectangle(zoomed, (round(centerx-8),round(centery-8)), (round(centerx+8),round(centery+8)), (40,40,255),2)
    result['outputs'].append(output('Magnified selected pixel', zoomed, f'Red box marks the pixel at ({x}, {y})'))
    result['stats'].extend([{'label':'Selected pixel','value':f'({x}, {y})'}, {'label':'Original RGB','value':f'{r}, {g}, {b}'}, {'label':'Total pixels','value':f'{w*h:,}'}])
    return result
