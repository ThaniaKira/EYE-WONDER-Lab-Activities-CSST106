"""Lab 06 — YOLO11 pre-trained detection on images and videos."""
from __future__ import annotations
import os
import uuid
import tempfile
from collections import Counter
from pathlib import Path
import cv2
from .common import base_result, output, num

_MODEL_CACHE={}

def get_model():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ValueError('YOLO11 requires Ultralytics. Install the optional requirements-advanced.txt file first.') from exc
    weights=os.getenv('VISIONLAB_YOLO_WEIGHTS','yolo11n.pt')
    if weights not in _MODEL_CACHE:
        try:
            _MODEL_CACHE[weights]=YOLO(weights)
        except Exception as exc:
            raise ValueError('Could not load YOLO11 weights. The first pretrained run may need internet access to download yolo11n.pt. '+str(exc)[:180]) from exc
    return _MODEL_CACHE[weights],weights


def process(img,form,files=None):
    model,weights=get_model()
    confidence=num(form,'confidence',0.35,0.05,0.95)
    preds=model.predict(source=img,conf=confidence,imgsz=640,verbose=False)
    pred=preds[0]
    annotated=pred.plot()
    result=base_result(img,'YOLO11 · Object Detection',note='YOLO11n pretrained inference. Bounding boxes, predicted labels, and confidence are shown. Custom dataset training/validation is a separate required exercise; document your actual trained model and metrics rather than treating pretrained inference as proof of training.')
    result['outputs'].append(output('Detected objects · YOLO11',annotated))
    classes=model.names
    counts=Counter(classes[int(i)] for i in pred.boxes.cls.cpu().tolist()) if pred.boxes is not None else Counter()
    result['stats'].extend([{'label':'Model weights','value':Path(weights).name},{'label':'Total detections','value':str(sum(counts.values()))},{'label':'Classes found','value':str(len(counts))}])
    if counts:
        result['note']+=' Counts: '+', '.join(f'{k}: {v}' for k,v in counts.most_common(12))
    return result


def process_video(file_storage, form, media_dir):
    model,weights=get_model()
    confidence=num(form,'confidence',.35,.05,.95)
    max_frames=180
    with tempfile.NamedTemporaryFile(suffix='.mp4',delete=False) as tmp:
        tmp.write(file_storage.read())
        src=tmp.name
    capture=cv2.VideoCapture(src)
    writer=None
    try:
        if not capture.isOpened():
            raise ValueError('The uploaded video could not be decoded. Try a short MP4 clip.')
        fps=capture.get(cv2.CAP_PROP_FPS)
        if not fps or fps < 1 or fps>60:fps=24
        name=uuid.uuid4().hex+'.mp4'
        out_path=Path(media_dir)/name
        counts=Counter(); n=0
        while n<max_frames:
            ok,frame=capture.read()
            if not ok:break
            h,w=frame.shape[:2]
            if max(h,w)>720:
                s=720/max(h,w);frame=cv2.resize(frame,(int(w*s)//2*2,int(h*s)//2*2))
            # Most MP4 codecs require even dimensions.
            hh,ww=frame.shape[:2]
            if ww%2 or hh%2:
                frame=cv2.resize(frame,(max(2,ww-ww%2),max(2,hh-hh%2)))
            if writer is None:
                hh,ww=frame.shape[:2]
                writer=cv2.VideoWriter(str(out_path),cv2.VideoWriter_fourcc(*'mp4v'),fps,(ww,hh))
                if not writer.isOpened():
                    raise ValueError('OpenCV could not create the output video on this machine.')
            pred=model.predict(source=frame,conf=confidence,imgsz=640,verbose=False)[0]
            if pred.boxes is not None:
                counts.update(model.names[int(v)] for v in pred.boxes.cls.cpu().tolist())
            writer.write(pred.plot())
            n+=1
        if not n: raise ValueError('No frames were read from this video.')
        return {'title':'YOLO11 · Video Detection','outputs':[],'video':f'/media/{name}','stats':[{'label':'Processed frames','value':str(n)},{'label':'Model','value':Path(weights).name},{'label':'Classes found','value':str(len(counts))}], 'note':'Processed up to 180 frames to keep the class demo responsive. Counts represent detections across frames, not unique tracked objects. '+'; '.join(f'{k}: {v}' for k,v in counts.most_common(10))}
    finally:
        capture.release()
        if writer is not None:writer.release()
        try:os.unlink(src)
        except OSError:pass
