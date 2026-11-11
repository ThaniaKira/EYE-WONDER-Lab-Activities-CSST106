"""Train and validate your own YOLO11 object detector on an annotated YOLO dataset.

Example:
  python scripts/train_yolo.py --data datasets/objects/data.yaml --epochs 50
Requires: pip install -r requirements-advanced.txt

The input data.yaml must reference real train and val image/label folders.
For meaningful final evaluation, include a held-out test split in data.yaml.
"""
import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',required=True,help='Path to your annotated dataset YAML file')
    parser.add_argument('--epochs',type=int,default=50)
    parser.add_argument('--imgsz',type=int,default=640)
    parser.add_argument('--model',default='yolo11n.pt')
    parser.add_argument('--test',action='store_true',help='Evaluate on a separate test split')
    args=parser.parse_args()
    if not Path(args.data).is_file():
        parser.error('Dataset YAML not found. Create a real dataset and data.yaml first.')
    from ultralytics import YOLO
    model=YOLO(args.model)
    training=model.train(data=args.data,epochs=args.epochs,imgsz=args.imgsz,project='runs/visionlab',name='custom-detection',exist_ok=False)
    weights=Path(model.trainer.best)
    best=YOLO(str(weights))
    val=best.val(data=args.data,split='val')
    metrics={'weights':str(weights),'validation_precision':float(val.box.mp),'validation_recall':float(val.box.mr),'validation_mAP50':float(val.box.map50),'validation_mAP50_95':float(val.box.map)}
    if args.test:
        test=best.val(data=args.data,split='test')
        metrics['test_precision']=float(test.box.mp)
        metrics['test_recall']=float(test.box.mr)
        metrics['test_mAP50']=float(test.box.map50)
        metrics['test_mAP50_95']=float(test.box.map)
    metrics_path=weights.parent.parent/'visionlab_metrics.json'
    metrics_path.write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    print('Metrics saved:',metrics_path)
    print('To load custom weights in the Web UI, set VISIONLAB_YOLO_WEIGHTS to:',weights)

if __name__=='__main__':main()
