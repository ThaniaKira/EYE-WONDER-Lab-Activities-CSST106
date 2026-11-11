"""Render standalone static design-preview HTML pages without running Flask.
The interactive processing buttons work only while the real Flask server is running.
"""
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, select_autoescape

root=Path(__file__).resolve().parents[1]
# Do not import app.py: this preview should generate without Flask being installed.
LABS=[
(1,'Image Manipulation','Image Manipulation','INTRODUCTION TO COMPUTER VISION','Explore pixels, grayscale conversion, resizing, cropping, and image rotations. The building blocks of digital vision.','image','pink',['OpenCV','Pixels','Color spaces'],'Starter lab'),
(2,'Edge Detection','Sobel & Canny Edges','FEATURE EXTRACTION','Make the invisible visible. Trace object boundaries and compare the gradient-based Sobel and threshold-based Canny operators.','scan','lilac',['Sobel','Canny','Gaussian blur'],'Image analysis'),
(3,'Histogram & Contrast','Histogram & Contrast','IMAGE ENHANCEMENT','Brighten hidden details, explore intensity distributions and compare global equalization with CLAHE.','chart','peach',['Histograms','CLAHE','Equalization'],'Image analysis'),
(4,'Feature Detection','SURF, SIFT & ORB','KEYPOINTS AND DESCRIPTORS','Find distinctive corners and features; match two images and discover how local feature descriptors work.','focus','mint',['SURF','SIFT','ORB'],'Feature matching'),
(5,'PCA & Eigenfaces','PCA & Eigenfaces','DIMENSIONALITY REDUCTION','Compress images using principal components, or run the assigned Eigenfaces and SVM recognition experiment with LFW.','face','lavender',['PCA','Eigenfaces','SVM'],'Machine learning'),
(6,'YOLO Detection','YOLO11 Object Detection','REAL-TIME COMPUTER VISION','Locate objects with bounding boxes, labels and confidence scores using pretrained YOLO11, on photos or short videos.','box','blue',['YOLO11','Inference','Bounding boxes'],'Deep learning'),
(7,'Segmentation','Image Segmentation','UNDERSTANDING PIXELS','Separate scenes into regions using thresholding, region growing, watershed, edge-based and semantic segmentation.','layers','rose',['Otsu','Watershed','DeepLabV3'],'Advanced vision'),
]
labs=[dict(zip(('id','short','title','eyebrow','description','icon','theme','tags','time'),row)) for row in LABS]
def url_for(endpoint,**kwargs):
    if endpoint=='static':return 'static/'+kwargs['filename']
    if endpoint=='dashboard':return 'preview.html'
    if endpoint=='guide':return 'preview-guide.html'
    if endpoint=='laboratory':return f"preview-lab{int(kwargs['lab_id']):02d}.html"
    raise RuntimeError(endpoint)
env=Environment(loader=FileSystemLoader(root/'templates'),autoescape=select_autoescape(['html']))
env.globals['url_for']=url_for
for filename,template,context in [('preview.html','dashboard.html',{'page':'dashboard'}),('preview-guide.html','guide.html',{'page':'guide'})]+[(f'preview-lab{lab["id"]:02d}.html','lab.html',{'page':'lab','lab':lab}) for lab in labs]:
    (root/filename).write_text(env.get_template(template).render(labs=labs,**context),encoding='utf-8')
    print('Created',filename)
