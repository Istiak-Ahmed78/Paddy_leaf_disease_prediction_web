import uvicorn
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import sklearn
import numpy as np
from io import BytesIO
from skimage.transform import resize
from skimage.io import imread
import pickle
import imageio as iio
import asyncio
 
# read an image 
img = iio.imread("download.jpeg")
img_resize=resize(img,(150,150,3))
l=[img_resize.flatten()]
def predict():
    
    r=load_model().predict(l)
    rp=load_model().predict_proba(l)
    classes = ['Leaf bright','Brown spot','Leaf smut']
    print({
        "Predicted": classes[r[0]],
        "Confidance": np.max(rp[0])
    })
    return {
        "Predicted": classes[r[0]],
        "Confidance": np.max(rp[0])
    }

def resize_image(data):
    img = np.array(Image.open(BytesIO(data)))
    img_resize=resize(img,(150,150,3))
    l=[img_resize.flatten()]
    return l

def load_model():
    filename = 'peddy_lead_svm.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model

r = predict()
print(r)