from io import BytesIO
from fastapi.testclient import TestClient
from PIL import Image
from backend.main import app
from backend import main

client=TestClient(app)
def test_health():
    assert client.get('/api/health').json()['status']=='ok'
def test_missing_input():
    assert client.post('/api/search').status_code==422
def test_invalid_image():
    assert client.post('/api/search',files={'image':('bad.jpg',b'not-an-image','image/jpeg')}).status_code==400
def test_oversized_upload():
    assert client.post('/api/search',files={'image':('big.jpg',b'x'*(main.MAX_BYTES+1),'image/jpeg')}).status_code==413
def test_image_and_text_flow(monkeypatch):
    seen={}
    def search(text,image,limit):
        seen.update(text=text,size=image.size,limit=limit)
        return [{'id':'shirt.jpg','title':'shirt','image_url':'/catalog/shirt.jpg','similarity':0.5}]
    monkeypatch.setattr(main.engine,'search',search)
    monkeypatch.setattr(main.engine,'classify',lambda image:None)
    raw=BytesIO();Image.new('RGB',(20,30)).save(raw,format='PNG')
    r=client.post('/api/search',data={'text':'blue shirt','limit':3},files={'image':('shirt.png',raw.getvalue(),'image/png')})
    assert r.status_code==200
    assert r.json()['matches'][0]['similarity']==0.5
    assert seen=={'text':'blue shirt','size':(20,30),'limit':3}
def test_empty_catalog_without_model_download():
    assert client.post('/api/search',data={'text':'shirt'}).json()['matches']==[]
def test_bad_limit():
    assert client.post('/api/search',data={'text':'shirt','limit':50}).status_code==422
