"""Train a supervised classifier on frozen CLIP embeddings; never train on val/test."""
import argparse, hashlib, json
from pathlib import Path
import joblib
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from backend.engine import Engine, MODEL_NAME, ROOT

EXT={'.jpg','.jpeg','.png','.webp'}
def collect(folder):
    paths=sorted(p for p in folder.glob('*/*') if p.is_file() and p.suffix.lower() in EXT)
    if not paths:
        raise ValueError(f'No labeled images in {folder}. Expected class_name/image.jpg.')
    return paths, np.array([p.parent.name for p in paths])
def fingerprints(paths):
    return {hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
def embed(engine,paths):
    vectors=[]
    for path in paths:
        with Image.open(path) as im:
            vectors.append(engine.encode([im.convert('RGB')])[0])
    return np.stack(vectors)
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,default=ROOT/'data')
    parser.add_argument('--test',action='store_true',help='Run a final test evaluation after model selection.')
    args=parser.parse_args()
    train,y=collect(args.data/'train'); val,yv=collect(args.data/'val')
    if len(set(y))<2: raise ValueError('Training needs at least two clothing classes.')
    if set(yv)!=set(y): raise ValueError('Train and validation must contain the same class names.')
    if fingerprints(train)&fingerprints(val): raise ValueError('Duplicate files across train and validation. Remove leakage.')
    engine=Engine(); x=embed(engine,train); xv=embed(engine,val)
    trials=[]; best=None
    for c in [0.1,1.0,10.0]:
        model=LogisticRegression(C=c,class_weight='balanced',max_iter=2000,random_state=42).fit(x,y)
        pred=model.predict(xv); score=f1_score(yv,pred,average='macro',zero_division=0)
        trials.append({'C':c,'validation_macro_f1':float(score)})
        if best is None or score>best[0]:best=(score,model)
    score,model=best
    metrics={'encoder':MODEL_NAME,'training':'Logistic regression on frozen CLIP embeddings; CLIP is not fine-tuned.',
             'train_images':len(train),'validation_images':len(val),'classes':model.classes_.tolist(),'trials':trials,
             'validation':classification_report(yv,model.predict(xv),output_dict=True,zero_division=0),
             'majority_baseline_accuracy':float(np.mean(yv==max(set(y),key=lambda label:sum(y==label))))}
    if args.test:
        test,yt=collect(args.data/'test')
        if set(yt)!=set(y):raise ValueError('Test must contain the same class names.')
        if fingerprints(test)&(fingerprints(train)|fingerprints(val)):raise ValueError('Test duplicates detected.')
        metrics['test']=classification_report(yt,model.predict(embed(engine,test)),output_dict=True,zero_division=0)
    (ROOT/'models').mkdir(exist_ok=True)
    joblib.dump({'encoder':MODEL_NAME,'model':model},ROOT/'models/classifier.joblib')
    (ROOT/'models/metrics.json').write_text(json.dumps(metrics,indent=2))
    print(json.dumps(metrics,indent=2))
if __name__=='__main__':main()
