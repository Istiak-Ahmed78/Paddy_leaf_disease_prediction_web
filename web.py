import uvicorn
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import sklearn
import numpy as np
from io import BytesIO
from skimage.transform import resize
from skimage.io import imread
import pickle

food_items = {
    'indian' : [ "Samosa", "Dosa" ],
    'american' : [ "Hot Dog", "Apple Pie"],
    'italian' : [ "Ravioli", "Pizza"]
}

app = FastAPI()
@app.get('/ping')
async def re():
    return 'Meo'

@app.post('/predict')
async def predict(file: UploadFile):
    contents =await file.read()
    conv = resize_image(contents)
    r=load_model().predict(conv)
    classes = ['Leaf bright','Brown spot','Leaf smut']
    print(r)
    return classes[r[0]]

def resize_image(data):
    img = np.array(Image.open(BytesIO(data)))
    img_resize=resize(img,(150,150,3))
    l=[img_resize.flatten()]
    return l

def load_model():
    filename = 'peddy_lead_svm.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model

if __name__=='__main__':
    uvicorn.run(app,host='localhost',port=8000)