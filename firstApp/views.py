from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os
# Create your views here.

model = load_model('./models/alexnet.h5')

def index(request):
    context = {'a': 1}
    return render(request, 'index.html', context)

def predictImage(request):
    fileObj = request.FILES['filePath']
    fs = FileSystemStorage()
    fs.save(fileObj.name, fileObj)
    filePathName = fs.url(fs.save(fileObj.name, fileObj))
    testimage = '.'+filePathName
    img = image.load_img(testimage, target_size=(227,227))
    img = image.img_to_array(img)/255
    img = np.expand_dims(img, axis=0)

    result = model.predict(img)
    predicted_class = np.argmax(result)
    confidence = round(result[0][predicted_class]*100, 2)
    if predicted_class == 0:
        predictedLabel = 'Cat'
    elif predicted_class == 1:
        predictedLabel = 'Dog'
    else:
        predictedLabel = 'Neither'


    context = {'filePathName': filePathName, 'predictedLabel': predictedLabel, 'confidence': confidence}
    return render(request, 'index.html', context)

def viewDataBase(request):
    images = os.listdir('./media/')
    imagesPaths = ['./media/'+i for i in images]
    context = {'imagesPaths': imagesPaths}
    return render(request, 'viewDB.html', context)