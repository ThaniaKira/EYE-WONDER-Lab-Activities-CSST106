"""Lab 05 — PCA image exploration or LFW Eigenfaces + SVM lab experiment."""
from __future__ import annotations
import cv2
import numpy as np
from .common import base_result, output, num


def line_chart(values, title='Cumulative explained variance', threshold=.90):
    w,h=800,300
    bg=np.full((h,w,3),(251,246,255),dtype=np.uint8)
    ml, mr, mt, mb = 68,27,38,46
    pw,ph=w-ml-mr,h-mt-mb
    for z in range(0,11,2):
        y=mt+ph-round(z/10*ph)
        cv2.line(bg,(ml,y),(w-mr,y),(228,216,239),1,cv2.LINE_AA)
        cv2.putText(bg,f'{z*10}%',(18,y+5),cv2.FONT_HERSHEY_SIMPLEX,.42,(109,79,135),1,cv2.LINE_AA)
    cv2.line(bg,(ml,mt+round((1-threshold)*ph)),(w-mr,mt+round((1-threshold)*ph)),(108,155,229),1,cv2.LINE_AA)
    if len(values):
        arr=np.asarray(values).flatten()
        xs=ml+np.arange(len(arr))*pw/max(1,len(arr)-1)
        ys=mt+ph*(1-np.clip(arr,0,1))
        pts=np.column_stack((xs,ys)).astype(np.int32)
        cv2.polylines(bg,[pts],False,(174,76,184),3,cv2.LINE_AA)
    cv2.putText(bg,title,(ml,22),cv2.FONT_HERSHEY_SIMPLEX,.60,(73,41,92),2,cv2.LINE_AA)
    for x in range(0,5):
        xx=ml+round(x*pw/4)
        cv2.putText(bg,str(1+int(x*max(0,len(values)-1)/4)),(xx-8,h-16),cv2.FONT_HERSHEY_SIMPLEX,.42,(109,79,135),1,cv2.LINE_AA)
    return bg


def face_gallery(images, labels=None, *, cell_w=120, cell_h=140, cols=4):
    if not images:
        return np.full((100,100,3),240,np.uint8)
    rows=(len(images)+cols-1)//cols
    canvas=np.full((rows*cell_h,cols*cell_w,3),(251,246,255),np.uint8)
    for i, item in enumerate(images):
        face = cv2.resize(np.uint8(np.clip(item,0,255)),(cell_w-12,cell_h-32))
        if face.ndim==2: face=cv2.cvtColor(face,cv2.COLOR_GRAY2BGR)
        r,c=divmod(i,cols)
        canvas[r*cell_h+4:r*cell_h+cell_h-28,c*cell_w+6:(c+1)*cell_w-6]=face
        lbl=(labels or [])[i] if labels is not None and i<len(labels) else f'Component {i+1}'
        cv2.putText(canvas,lbl[:19],(c*cell_w+6,(r+1)*cell_h-11),cv2.FONT_HERSHEY_SIMPLEX,.33,(78,45,105),1,cv2.LINE_AA)
    return canvas


def image_pca(img,form):
    """Supplementary single-image PCA compression demo; not an Eigenfaces classifier."""
    maxside=380
    h,w=img.shape[:2]
    if max(h,w)>maxside:
        scale=maxside/max(h,w)
        img=cv2.resize(img,(round(w*scale),round(h*scale)))
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    n=num(form,'components',28,2,min(gray.shape)-1,int)
    from sklearn.decomposition import PCA
    pca=PCA(n_components=n,svd_solver='randomized',random_state=42)
    reduced=pca.fit_transform(gray.astype(np.float32))
    restored=np.clip(pca.inverse_transform(reduced),0,255).astype(np.uint8)
    variance=np.cumsum(pca.explained_variance_ratio_)
    result=base_result(img,'PCA Image Compression',note='Supplementary image-compression experiment: PCA here reduces rows of one image. For the assigned Eigenfaces face-recognition experiment, select “LFW · Eigenfaces + SVM”.')
    result['outputs'] += [output(f'Reconstructed · {n} components',restored),output('Cumulative explained variance',line_chart(variance))]
    result['stats'] += [{'label':'Components','value':str(n)},{'label':'Variance retained','value':f'{variance[-1]*100:.1f}%'},{'label':'Image MSE','value':f'{np.mean((gray.astype(float)-restored.astype(float))**2):.1f}'}]
    return result


def lfw_eigenfaces(form):
    from sklearn.datasets import fetch_lfw_people
    from sklearn.model_selection import train_test_split
    from sklearn.decomposition import PCA
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    try:
        lfw=fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    except Exception as exc:
        raise ValueError('LFW face dataset is not cached and could not be downloaded. Connect to the internet for the first LFW run, then try again. Error: '+str(exc)[:180]) from exc
    n_samples,h,w=lfw.images.shape
    X_train,X_test,y_train,y_test=train_test_split(lfw.data,lfw.target,test_size=.2,random_state=42)
    n=int(num(form,'components',150,5,min(150,X_train.shape[0]-1,X_train.shape[1]),int))
    pca=PCA(n_components=n,svd_solver='randomized',whiten=True,random_state=42)
    train=pca.fit_transform(X_train)
    test=pca.transform(X_test)
    svm=SVC(kernel='rbf',class_weight='balanced')
    svm.fit(train,y_train)
    pred=svm.predict(test)
    precision,recall,f1,_=precision_recall_fscore_support(y_test,pred,average='weighted',zero_division=0)
    eigens=pca.components_.reshape(n,h,w)
    eigenimgs=[cv2.normalize(e,None,0,255,cv2.NORM_MINMAX) for e in eigens[:12]]
    wrong=np.where(pred!=y_test)[0][:8]
    failures=[X_test[k].reshape(h,w) for k in wrong]
    labels=[f'P:{lfw.target_names[pred[k]][:12]}' for k in wrong]
    variance=np.cumsum(pca.explained_variance_ratio_)
    ninety=np.where(variance>=.90)[0]
    n90=f'{ninety[0]+1}' if len(ninety) else f'> {n} (not reached)'
    result={'title':'LFW · Eigenfaces and SVM','outputs':[output('First 12 eigenfaces',face_gallery(eigenimgs)), output('Cumulative explained variance',line_chart(variance))], 'stats':[{'label':'LFW images','value':str(n_samples)},{'label':'Recognized identities','value':str(len(lfw.target_names))},{'label':'SVM accuracy','value':f'{accuracy_score(y_test,pred)*100:.1f}%'},{'label':'Weighted precision','value':f'{precision*100:.1f}%'},{'label':'Weighted recall','value':f'{recall*100:.1f}%'},{'label':'Weighted F1','value':f'{f1*100:.1f}%'},{'label':'Components for ≥90%','value':n90}], 'note':'Assignment workflow: LFW min_faces_per_person=70, 80/20 train/test split, randomized PCA with whitening, and balanced RBF SVM. The 90% threshold is reported only if reached in the fitted components. Faces are evaluated on held-out test images.'}
    if failures:
        result['outputs'].append(output('Incorrect predictions · failure cases',face_gallery(failures,labels,cols=4),'Analyze lighting, pose, facial occlusions and accessories.'))
    else:
        result['note']+=' No incorrect predictions were found in this evaluation.'
    result['outputs'].append(output('Last 3 eigenfaces',face_gallery([cv2.normalize(e,None,0,255,cv2.NORM_MINMAX) for e in eigens[-3:]],labels=[f'Component {n-2+i}' for i in range(3)],cols=3)))
    return result


def process(img,form,files=None):
    if form.get('mode')=='faces':
        return lfw_eigenfaces(form)
    if img is None:
        raise ValueError('Upload an image for PCA compression, or choose the LFW Eigenfaces + SVM experiment.')
    return image_pca(img,form)
