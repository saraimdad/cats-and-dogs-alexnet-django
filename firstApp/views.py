from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import os
from django.http import JsonResponse
from django.views.decorators.http import require_GET, require_POST
from PIL import Image
import requests
from io import BytesIO

# Construct the absolute path to the model file
current_directory = os.path.dirname(__file__)
model_path = os.path.join(current_directory, 'alexnet.h5')

# Load the model
model = load_model(model_path)

def index(request):
    context = {'a': 1}
    return render(request, 'index.html', context)

def predictImage(request):
    if 'filePath' in request.FILES:
        fileObj = request.FILES['filePath']
        return predict_image_from_file(request, fileObj, render_template=True)
   
    return render(request, 'index.html')
    

def viewDataBase(request):
    images = os.listdir('./media/')
    imagesPaths = ['./media/' + i for i in images]
    context = {'imagesPaths': imagesPaths}
    return render(request, 'viewDB.html', context)

@require_GET
def predict_image_from_url(request):
    image_url = request.GET.get('image_url')
    if not image_url:
        return JsonResponse({"error": "No image URL provided"}, status=400)
    
    try:
        response = requests.get(image_url)
        response.raise_for_status()  
        img = Image.open(BytesIO(response.content))  
        img = img.resize((227, 227))
        img = img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        result = model.predict(img)
        predicted_class = np.argmax(result)
        confidence = round(result[0][predicted_class] * 100, 2)
        
        if predicted_class == 0:
            predictedLabel = 'Cat'
        elif predicted_class == 1:
            predictedLabel = 'Dog'
        else:
            predictedLabel = 'Neither'

        return JsonResponse({
            'predictedLabel': predictedLabel,
            'confidence': confidence
        })
    except requests.exceptions.RequestException as e:
        return JsonResponse({"error": "Error fetching the image: " + str(e)}, status=500)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

@require_POST
def predict_image_from_file(request, fileObj=None, render_template=False):
    if fileObj is None:
        fileObj = request.FILES.get('file')
        if not fileObj:
            return JsonResponse({"error": "No file uploaded"}, status=400)
    
    fs = FileSystemStorage()
    filename = fs.save(fileObj.name, fileObj)
    filePathName = fs.url(filename)
    testimage = '.' + filePathName
    
    try:
        img = load_img(testimage, target_size=(227, 227))
        img = img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        result = model.predict(img)
        predicted_class = np.argmax(result)
        confidence = round(result[0][predicted_class] * 100, 2)
        
        if predicted_class == 0:
            predictedLabel = 'Cat'
        elif predicted_class == 1:
            predictedLabel = 'Dog'
        else:
            predictedLabel = 'Neither'
        
        if render_template:
            context = {'filePathName': filePathName, 'predictedLabel': predictedLabel, 'confidence': confidence}
            return render(request, 'index.html', context)
        else:
            return JsonResponse({
                'predictedLabel': predictedLabel,
                'confidence': confidence
            })
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
