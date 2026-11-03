from io import BytesIO
from pathlib import Path
import warnings
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from backend.engine import Engine, ROOT

app=FastAPI(title='Fashion Lens',version='0.1.0')
app.add_middleware(CORSMiddleware,allow_origins=['http://localhost:5173','http://127.0.0.1:5173'],allow_methods=['GET','POST'],allow_headers=['*'])
app.mount('/catalog',StaticFiles(directory=ROOT/'data/catalog'),name='catalog')
engine=Engine()
MAX_BYTES=8*1024*1024

def decode_image(raw):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error',Image.DecompressionBombWarning)
            with Image.open(BytesIO(raw)) as im:
                if im.format not in {'JPEG','PNG','WEBP'}:
                    raise HTTPException(415,'Use a JPG, PNG, or WebP image.')
                from PIL import ImageOps
                return ImageOps.exif_transpose(im).convert('RGB').copy()
    except (UnidentifiedImageError,OSError,Image.DecompressionBombError,Image.DecompressionBombWarning):
        raise HTTPException(400,'This image could not be read. Try a smaller JPG or PNG.')

@app.get('/api/health')
def health():
    count=sum(p.suffix.lower() in {'.jpg','.jpeg','.png','.webp'} for p in (ROOT/'data/catalog').iterdir())
    return {'status':'ok','catalog_count':count,'encoder':'Pretrained CLIP','encoder_loaded':engine._model is not None,
            'classifier_trained':(ROOT/'models/classifier.joblib').exists()}

@app.post('/api/search')
def search(text:str=Form(''),limit:int=Form(12),image:UploadFile|None=File(None)):
    if not 1<=limit<=30:
        raise HTTPException(422,'Result limit must be between 1 and 30.')
    if len(text)>500:
        raise HTTPException(422,'Keep the description under 500 characters.')
    photo=None
    if image is not None:
        raw=image.file.read(MAX_BYTES+1)
        if len(raw)>MAX_BYTES:
            raise HTTPException(413,'Choose an image smaller than 8 MB.')
        photo=decode_image(raw)
    if photo is None and not text.strip():
        raise HTTPException(422,'Upload a photo or describe an item.')
    try:
        matches=engine.search(text,photo,limit)
        classification=engine.classify(photo) if photo is not None and matches else None
        return {'matches':matches,'classification':classification,
                'method':'Pretrained CLIP cosine similarity. Similarity is not an accuracy percentage.',
                'message':'Add clothing images to data/catalog, then search again.' if not matches else None}
    except HTTPException:
        raise
    except Exception:
        import logging
        logging.exception('Search failed')
        raise HTTPException(503,'AI could not finish. Check the backend terminal; first use needs an internet connection to download CLIP.')
