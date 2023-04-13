from django.shortcuts import render
from PIL import Image
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.applications.vgg19 import preprocess_input
import numpy as np

model=load_model("best_model.h5")

result = {
  0 : "Apple_Apple_scab",
  1 : "Apple_Black_rot",
  2 :  "Apple_Cedar_apple_rust",
  3 : "Apple_healthy",
  4 : "Corn_cercospora_leaf_spot_Gray_leaf_spot",
  5 : "Corn_Common_rust",
  6 : "Corn_healthy",
  7 : "Corn_Northern_leaf_Blight",
  8 : "Potato_Early_blight",
  9 : "Potato_healthy",
  10 : "Potato_Late_blight",
  11 : "Tomato_Bacterial_spot",
  12 : "Tomato_early_blight",
  13 : "Tomato_healthy",
  14 : "Tomato_Late_blight",
  15 : "Tomato_Leaf_mold",
  16 : "Tomato_septoria_leaf_spot",
  17 : "Tomato_spider_mites Two-spotted_spider_mite",
  18 : "Tomato_target_spot",
  19 : "Tomato_mosaic_virus",
  20 : "Tomato_tomato_yellow_leaf_curl_virus"
}

def home(request):
  return render(request, 'home.html')

def upload(request):
  context = {}
  if request.method == "POST":
    if 'img' in request.FILES:
      img = request.FILES['img']
      imag = Image.open(img)              #Converts input image to PIL

      open_cv_image = np.array(imag)
      open_cv_image = open_cv_image[:, :, ::-1].copy()    #Convert PIL to cv2
      cv2.imshow("hi",open_cv_image)

      imag = cv2.resize(open_cv_image, (256, 256))      #Resize cv2 image
      print("Final = ",imag)
      cv2.imshow('image', imag)

      color_coverted = cv2.cvtColor(imag, cv2.COLOR_BGR2RGB)
      pil_image = Image.fromarray(color_coverted)       #Converts resized cv2 image again to PIL
      pil_image.show()

      imagR=img_to_array(pil_image)

      im=preprocess_input(imagR)
      img=np.expand_dims(im,axis=0)
      img=img.reshape(1,256,256,3)
      pred=np.argmax(model.predict(img))
      context = {
        'pred' : result[pred],
        }
  return render(request, 'upload.html', context)

def supplements(request):
  return render(request, 'supplements.html')