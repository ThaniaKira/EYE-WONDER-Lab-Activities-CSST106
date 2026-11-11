"""Lab 07 — global/adaptive, region-growing, watershed, contour, and semantic segmentation."""
from collections import deque
import cv2
import numpy as np
from .common import base_result, output, num


def region_growing(gray, x, y, tolerance):
    # Work at reduced resolution to keep bounded CPU and memory for classroom demo.
    h,w=gray.shape
    result=np.zeros((h,w),np.uint8)
    seen=np.zeros((h,w),np.uint8)
    q=deque([(x,y)])
    ref=int(gray[y,x]); seen[y,x]=1
    while q:
        a,b=q.popleft()
        if abs(int(gray[b,a])-ref)>tolerance:
            continue
        result[b,a]=255
        for nx,ny in ((a-1,b),(a+1,b),(a,b-1),(a,b+1)):
            if 0<=nx<w and 0<=ny<h and not seen[ny,nx]:
                seen[ny,nx]=1
                q.append((nx,ny))
    return result


def watershed_overlay(img):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _,binary=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    kernel=np.ones((3,3),np.uint8)
    opening=cv2.morphologyEx(binary,cv2.MORPH_OPEN,kernel,iterations=2)
    sure_bg=cv2.dilate(opening,kernel,iterations=3)
    dist=cv2.distanceTransform(opening,cv2.DIST_L2,5)
    _,sure_fg=cv2.threshold(dist,.45*dist.max() if dist.max()>0 else 1,255,0)
    sure_fg=np.uint8(sure_fg)
    unknown=cv2.subtract(sure_bg,sure_fg)
    count,markers=cv2.connectedComponents(sure_fg)
    markers=markers+1; markers[unknown==255]=0
    markers=cv2.watershed(img.copy(),markers)
    out=img.copy(); out[markers==-1]=(207,45,222)
    for idx in range(2,count+1):
        mask=markers==idx
        if np.count_nonzero(mask)>30:
            rgb=np.array([(idx*97)%165+60,(idx*53)%155+55,(idx*151)%135+70],dtype=np.float64)
            out[mask]=np.uint8(out[mask].astype(np.float64)*.55+rgb*.45)
    return out,max(0,count-1)


def semantic_deeplab(img):
    try:
        import torch
        from PIL import Image
        from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
    except Exception as exc:
        raise ValueError('Semantic DeepLabV3 requires the optional packages torch and torchvision. Install requirements-advanced.txt, then try again.') from exc
    weights=DeepLabV3_ResNet50_Weights.DEFAULT
    try:
        model=deeplabv3_resnet50(weights=weights).eval()
    except Exception as exc:
        raise ValueError('Could not load the pretrained DeepLabV3 weights. They download on first use and require an internet connection. '+str(exc)[:140]) from exc
    h,w=img.shape[:2]
    rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    pil=Image.fromarray(rgb)
    tensor=weights.transforms()(pil).unsqueeze(0)
    with torch.inference_mode():
        pred=model(tensor)['out'][0].argmax(0).byte().cpu().numpy()
    pred=cv2.resize(pred,(w,h),interpolation=cv2.INTER_NEAREST)
    classes=weights.meta.get('categories') or ['background']
    palette=np.array([[0,0,0]]+[[((i*67)%180)+65,((i*131)%170)+70,((i*39)%170)+75] for i in range(1,256)],dtype=np.uint8)
    color=palette[pred]
    blended=cv2.addWeighted(img,.45,color,.55,0)
    visible=np.unique(pred)
    present=', '.join(f'{classes[int(i)] if int(i)<len(classes) else str(i)}' for i in visible if int(i)!=0)
    return blended,present or 'background only'


def process(img,form,files=None):
    method=form.get('seg_method','compare')
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    result=base_result(img,'Image Segmentation',note='Pixel/region methods produce foreground masks or region boundaries; semantic DeepLabV3 assigns class labels to pixels. The algorithm and threshold strongly affect results on textured or low-contrast scenes.')
    if method=='semantic':
        vis,labels=semantic_deeplab(img)
        result['outputs'].append(output('DeepLabV3 · semantic classes',vis,'Predicted foreground classes: '+labels))
        result['stats'].append({'label':'Predicted classes','value':labels[:100]})
        return result
    otsu_t,otsu=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    block=num(form,'block',15,3,51,int)
    if block%2==0:block+=1
    adaptive=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,block,2)
    canny=cv2.Canny(gray,70,165)
    contours,_=cv2.findContours(canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    mask=np.zeros_like(gray)
    cv2.drawContours(mask,[c for c in contours if cv2.contourArea(c)>120],-1,255,cv2.FILLED)
    ws,nregions=watershed_overlay(img)
    h,w=gray.shape
    # Process region-growing only at an appropriately sized resolution.
    sc=min(1,500/max(h,w))
    small=cv2.resize(gray,(max(1,round(w*sc)),max(1,round(h*sc)))) if sc<1 else gray
    sh,sw=small.shape
    sx=num(form,'seed_x',50,0,100,int)* (sw-1)//100
    sy=num(form,'seed_y',50,0,100,int)* (sh-1)//100
    tol=num(form,'tolerance',20,0,100,int)
    region=region_growing(small,sx,sy,tol)
    if sc<1: region=cv2.resize(region,(w,h),interpolation=cv2.INTER_NEAREST)
    if method in ('compare','threshold'):
        result['outputs'].extend([output('Otsu · global threshold',otsu),output('Adaptive threshold',adaptive)])
    if method in ('compare','region'):
        result['outputs'].append(output('Seeded region growing',region,f'Seed at {sx},{sy}; similarity tolerance {tol}'))
        result['outputs'].append(output('Watershed boundaries',ws,f'{nregions} candidate regions'))
    if method in ('compare','edge'):
        result['outputs'].append(output('Canny boundary edges',canny))
        result['outputs'].append(output('Closed contour segmentation',mask))
    result['stats'].extend([{'label':'Otsu threshold','value':f'{otsu_t:.0f}'},{'label':'Watershed regions','value':str(nregions)},{'label':'Region pixels','value':f'{cv2.countNonZero(region):,}'}])
    return result
