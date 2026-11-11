"""Lab 04 — SURF, SIFT and ORB comparison with real feature matching."""
import cv2
from .common import base_result, output, decode_image, num


def make_detector(name, hessian, upright):
    if name == 'SURF':
        try:
            return cv2.xfeatures2d.SURF_create(hessianThreshold=hessian, upright=upright)
        except (AttributeError, cv2.error) as exc:
            raise ValueError('SURF is not enabled in this OpenCV build (it often requires a separate nonfree-enabled build). Choose SIFT or ORB to explore alternative detectors; neither is labeled as SURF.') from exc
    if name == 'SIFT':
        if not hasattr(cv2, 'SIFT_create'):
            raise ValueError('SIFT is not available in this OpenCV installation.')
        return cv2.SIFT_create(nfeatures=900)
    return cv2.ORB_create(nfeatures=900)


def process(img, form, files=None):
    name = form.get('detector','SIFT').upper()
    if name not in ('SURF','SIFT','ORB'):
        raise ValueError('Select SURF, SIFT, or ORB.')
    hessian = num(form,'hessian',400,100,2000,int)
    upright = form.get('upright') == 'true'
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    detector = make_detector(name,hessian,upright)
    keypoints, descriptors = detector.detectAndCompute(gray,None)
    drawn = cv2.drawKeypoints(img,keypoints,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,color=(245,93,206))
    result = base_result(img,'Feature Detection',note=f'{name} features shown with size and orientation. Feature matching uses Lowe’s ratio test where descriptors are available. SURF is attempted only when explicitly selected.')
    result['outputs'].append(output(f'{name} keypoints',drawn))
    result['stats'].append({'label':'Detected keypoints','value':str(len(keypoints))})
    ref = (files or {}).get('reference')
    if ref and ref.filename:
        other = decode_image(ref, max_side=1200)
        other_gray = cv2.cvtColor(other, cv2.COLOR_BGR2GRAY)
        k2,d2 = detector.detectAndCompute(other_gray,None)
        if descriptors is None or d2 is None or len(descriptors)<2 or len(d2)<2:
            result['note'] += ' There are too few features to match between these two images.'
        else:
            norm = cv2.NORM_HAMMING if name == 'ORB' else cv2.NORM_L2
            matcher = cv2.BFMatcher(norm)
            possible = matcher.knnMatch(descriptors, d2, k=2)
            good = [a for pair in possible if len(pair)==2 for a,b in [pair] if a.distance < .75*b.distance]
            matched = cv2.drawMatches(img,keypoints,other,k2,sorted(good,key=lambda x:x.distance)[:65],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            result['outputs'].append(output(f'{name} image matching',matched,f'{len(good)} strong correspondences'))
            result['stats'].append({'label':'Good matches','value':str(len(good))})
    return result
