from django.shortcuts import render
from PIL import Image
from django.conf import settings
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.applications.vgg19 import preprocess_input
import numpy as np
import pandas as pd

model=load_model("best_model.h5")
disease_info = pd.read_csv(str(settings.BASE_DIR)+'\Crop_Cure\disease_info.csv' , encoding='cp1252')

result = {
  0 : "Apple_scab",
  1 : "Apple_Black_rot",
  2 : "Apple_Cedar_apple_rust",
  3 : "Apple_healthy",
  4 : "Corn_cercospora_leaf_spot_gray_leaf_spot",
  5 : "Corn_Common_rust",
  6 : "Corn_Northern_leaf_Blight",
  7 : "Corn_healthy",
  8 : "Potato_Early_blight",
  9 : "Potato_Late_blight",
  10 : "Potato_healthy",
  11 : "Tomato_Bacterial_spot",
  12 : "Tomato_early_blight",
  13 : "Tomato_Late_blight",
  14 : "Tomato_Leaf_mold",
  15 : "Tomato_septoria_leaf_spot",
  16 : "Tomato_spider_mites Two-spotted_spider_mite",
  17 : "Tomato_target_spot",
  18 : "Tomato_yellow_leaf_curl_virus",
  19 : "Tomato_mosaic_virus",
  20 : "Tomato_healthy"
}

def home(request):
  return render(request, 'home.html')

def upload(request):
  context = {}
  if request.method == "POST":
    if 'img' in request.FILES:
      img = request.FILES['img']
      imag = Image.open(img)              #Converts input image to PIL

      img_path = settings.MEDIA_ROOT + "\\input.jpeg"

      if imag.format=="PNG":              #If imge is PNG, it converts to JPEG
        imag = imag.convert("RGB")
        imag.save(img_path, "jpeg")

      iz = imag.resize((256,256))
      iz.save(img_path, "jpeg")

      imagR=img_to_array(iz)
      im=preprocess_input(imagR)
      img=np.expand_dims(im,axis=0)
      img=img.reshape(1,256,256,3)
      
      pred=np.argmax(model.predict(img))

      title = disease_info['disease_name'][pred]
      description =disease_info['description'][pred]
      prevent = disease_info['Possible Steps'][pred]
      image_url = disease_info['image_url'][pred]
      print("pred=", pred)
      context = {
        'pred' : pred,
        'result' : result[pred],
        'title' : title,
        'desc' : description,
        'prevent' : prevent,
        'image_url' : image_url
        }
      return render(request, 'result.html', context)
  return render(request, 'upload.html', context)

def supplements(request):
  return render(request, 'supplements.html')

def submit(request):
  return render(request, 'submit.html')