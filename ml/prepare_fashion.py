"""Download a reproducible, deduplicated subset for Fashion Lens classification."""
import argparse
from collections import defaultdict
import hashlib
import io
import json
from pathlib import Path
import random
import re
import shutil
from PIL import Image, ImageOps

DATASET='ashraq/fashion-product-images-small'
REVISION='main'
LABELS={'Shirts':'shirt','Tshirts':'tshirt','Jeans':'jeans','Dresses':'dress',
        'Casual Shoes':'shoes','Formal Shoes':'shoes','Sports Shoes':'shoes','Shorts':'shorts'}

def decode(value):
    if isinstance(value,dict):
        if value.get('bytes') is not None:
            return Image.open(io.BytesIO(value['bytes'])).convert('RGB')
        return Image.open(value['path']).convert('RGB')
    return value.convert('RGB')

def prepare(rows,out,per_class=500,seed=42):
    if out.exists():
        raise ValueError(f'{out} already exists. Use a new --output path to avoid mixing splits.')
    staging=out.with_name(out.name+'.partial')
    if staging.exists():
        raise ValueError(f'Incomplete run at {staging}. Rename it before retrying, or choose another --output.')
    staging.mkdir(parents=True)
    (staging/'pool').mkdir()
    selected=defaultdict(list);seen_hash=set();seen_ids=set();skipped=0
    try:
        for i,row in enumerate(rows):
            label=LABELS.get(row.get('articleType'))
            if label and len(selected[label])<per_class:
                try:
                    product_id=str(int(row['id']))
                    if product_id in seen_ids:continue
                    im=ImageOps.exif_transpose(decode(row['image']))
                    digest=hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()
                    if digest in seen_hash:continue
                    name=re.sub(r'\s+',' ',str(row.get('productDisplayName') or '').lower()).strip()
                    filename=f'{label}-{product_id}.png'
                    im.save(staging/'pool'/filename)
                    selected[label].append({'id':product_id,'label':label,'filename':filename,
                         'group':name or product_id,'pixel_sha256':digest,'source_articleType':row['articleType']})
                    seen_hash.add(digest);seen_ids.add(product_id)
                except (OSError,ValueError,KeyError):skipped+=1
            if i%1000==0:
                print(f'Scanned {i:,} rows; selected '+str({k:len(v) for k,v in selected.items()}),flush=True)
            if len(selected)==len(set(LABELS.values())) and all(len(v)>=per_class for v in selected.values()):break
        expected=set(LABELS.values())
        if set(selected)!=expected or any(len(selected[c])<30 for c in expected):
            raise ValueError('Insufficient images in one or more classes: '+str({k:len(v) for k,v in selected.items()}))
        rng=random.Random(seed);manifest=[];counts={}
        for label in sorted(selected):
            groups=defaultdict(list)
            for item in selected[label]:groups[item['group']].append(item)
            keys=sorted(groups);rng.shuffle(keys)
            if len(keys)<10:raise ValueError(f'Too few independent product groups for {label}')
            n=len(keys);a=int(n*.7);b=int(n*.85)
            counts[label]={}
            for split,part in [('train',keys[:a]),('val',keys[a:b]),('test',keys[b:])]:
                target=staging/split/label;target.mkdir(parents=True)
                count=0
                for key in part:
                    for item in groups[key]:
                        shutil.move(str(staging/'pool'/item['filename']),str(target/item['filename']))
                        manifest.append(dict(item,split=split,path=f'{split}/{label}/{item["filename"]}'));count+=1
                counts[label][split]=count
        (staging/'pool').rmdir()
        report={'dataset':DATASET,'revision':REVISION,'seed':seed,'per_class_cap':per_class,
                'counts':counts,'skipped_unreadable':skipped,
                'split_method':'70/15/15 by normalized product name within each class; exact pixel duplicates removed globally.',
                'limits':'Related products with different names and near-duplicates may remain. Low-resolution product photos differ from user photos.'}
        (staging/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        (staging/'dataset_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        staging.rename(out)
        print('READY: '+str(out.resolve()),flush=True)
        print(json.dumps(report,indent=2),flush=True)
        return report
    except Exception:
        print(f'Preparation stopped. Partial files are in {staging}; do not use them for training.',flush=True)
        raise

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--per-class',type=int,default=500)
    p.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'data/fashion_v1')
    args=p.parse_args()
    if not 30<=args.per_class<=5000:p.error('--per-class must be 30–5000')
    from datasets import load_dataset, Image as DatasetImage
    print('Downloading dataset from Hugging Face. First use needs internet.',flush=True)
    ds=load_dataset(DATASET,revision=REVISION,split='train',streaming=True)
    ds=ds.cast_column('image',DatasetImage(decode=False)).shuffle(seed=42,buffer_size=2000)
    prepare(ds,args.output,args.per_class)
if __name__=='__main__':main()
