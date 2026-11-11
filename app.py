"""VisionLab — CSST106 Computer Vision & Perception laboratory portfolio.
Run: python app.py, then open http://127.0.0.1:5000
"""
from __future__ import annotations
import os
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.exceptions import RequestEntityTooLarge
from modules.common import decode_image
from modules import lab01, lab02, lab03, lab04, lab05, lab06, lab07

BASE_DIR=Path(__file__).resolve().parent
MEDIA_DIR=BASE_DIR/'media'
MEDIA_DIR.mkdir(exist_ok=True)

app=Flask(__name__)
app.config['MAX_CONTENT_LENGTH']=32*1024*1024
app.config['SEND_FILE_MAX_AGE_DEFAULT']=3600

LABS=[
 {'id':1,'short':'Image Manipulation','title':'Image Manipulation','eyebrow':'INTRODUCTION TO COMPUTER VISION','description':'Explore pixels, grayscale conversion, resizing, cropping, and image rotations. The building blocks of digital vision.','icon':'image','theme':'pink','tags':['OpenCV','Pixels','Color spaces'], 'time':'Starter lab'},
 {'id':2,'short':'Edge Detection','title':'Sobel & Canny Edges','eyebrow':'FEATURE EXTRACTION','description':'Make the invisible visible. Trace object boundaries and compare the gradient-based Sobel and threshold-based Canny operators.','icon':'scan','theme':'lilac','tags':['Sobel','Canny','Gaussian blur'], 'time':'Image analysis'},
 {'id':3,'short':'Histogram & Contrast','title':'Histogram & Contrast','eyebrow':'IMAGE ENHANCEMENT','description':'Brighten hidden details, explore intensity distributions and compare global equalization with CLAHE.','icon':'chart','theme':'peach','tags':['Histograms','CLAHE','Equalization'], 'time':'Image analysis'},
 {'id':4,'short':'Feature Detection','title':'SURF, SIFT & ORB','eyebrow':'KEYPOINTS AND DESCRIPTORS','description':'Find distinctive corners and features; match two images and discover how local feature descriptors work.','icon':'focus','theme':'mint','tags':['SURF','SIFT','ORB'], 'time':'Feature matching'},
 {'id':5,'short':'PCA & Eigenfaces','title':'PCA & Eigenfaces','eyebrow':'DIMENSIONALITY REDUCTION','description':'Compress images using principal components, or run the assigned Eigenfaces and SVM recognition experiment with LFW.','icon':'face','theme':'lavender','tags':['PCA','Eigenfaces','SVM'], 'time':'Machine learning'},
 {'id':6,'short':'YOLO Detection','title':'YOLO11 Object Detection','eyebrow':'REAL-TIME COMPUTER VISION','description':'Locate objects with bounding boxes, labels and confidence scores using pretrained YOLO11, on photos or short videos.','icon':'box','theme':'blue','tags':['YOLO11','Inference','Bounding boxes'], 'time':'Deep learning'},
 {'id':7,'short':'Segmentation','title':'Image Segmentation','eyebrow':'UNDERSTANDING PIXELS','description':'Separate scenes into regions using thresholding, region growing, watershed, edge-based and semantic segmentation.','icon':'layers','theme':'rose','tags':['Otsu','Watershed','DeepLabV3'], 'time':'Advanced vision'},
]
LOOKUP={item['id']:item for item in LABS}
PROCESSORS={1:lab01.process,2:lab02.process,3:lab03.process,4:lab04.process,5:lab05.process,6:lab06.process,7:lab07.process}

@app.context_processor
def inject_global():
    return {'labs':LABS}

@app.get('/')
def dashboard():
    return render_template('dashboard.html',page='dashboard')

@app.get('/lab/<int:lab_id>')
def laboratory(lab_id):
    if lab_id not in LOOKUP:
        return render_template('not_found.html',page='404'),404
    return render_template('lab.html',page='lab',lab=LOOKUP[lab_id])

@app.get('/guide')
def guide():
    return render_template('guide.html',page='guide')

@app.post('/api/process/<int:lab_id>')
def process_lab(lab_id):
    if lab_id not in PROCESSORS:
        return jsonify({'error':'Unknown laboratory module.'}),404
    try:
        form=request.form
        if lab_id==6 and request.files.get('video') and request.files['video'].filename:
            result=lab06.process_video(request.files['video'],form,MEDIA_DIR)
        else:
            needs_image=not(lab_id==5 and form.get('mode')=='faces')
            img=decode_image(request.files.get('image')) if needs_image else None
            result=PROCESSORS[lab_id](img,form,request.files)
        return jsonify(result)
    except ValueError as exc:
        return jsonify({'error':str(exc)}),400
    except Exception as exc:
        app.logger.exception('Failed Lab %s processing',lab_id)
        return jsonify({'error':'Processing failed. Check your installation or inputs and retry. Technical detail: '+str(exc)[:170]}),500

@app.get('/media/<path:name>')
def media(name):
    # Only output mp4 filenames that this app generated; never allow arbitrary paths.
    if '/' in name or '\\' in name or not name.endswith('.mp4'):
        return jsonify({'error':'Invalid video path.'}),404
    return send_from_directory(MEDIA_DIR,name,mimetype='video/mp4')

@app.errorhandler(RequestEntityTooLarge)
def too_large(exc):
    if request.path.startswith('/api/'):
        return jsonify({'error':'File is too large. Maximum upload size is 32 MB.'}),413
    return 'Maximum upload size is 32 MB.',413

if __name__=='__main__':
    app.run(host=os.getenv('VISIONLAB_HOST','127.0.0.1'),port=int(os.getenv('PORT','5000')),debug=os.getenv('FLASK_DEBUG','0')=='1')
