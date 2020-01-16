import os
import shutil
import ntpath
import keras.models as models
from keras.preprocessing import image
import numpy as np


labels = ['charcoal-figure-standing', 'charcoal-landscape', 'charcoal-portrait', 'color-portrait-chromatic', 'color-portrait-dark-background', 'field-chromatic-small-horizon', 'landscape-chromatic', 'landscape-with-horizon', 'object-chromatic']

# we're classifying ata from vangogh-256
image_size = 256

model = models.load_model('vangogh-256-classifier-small_last4.h5')

batch_size = 16
batches = []
current_batch = []
index = -1
for root, dirs, files in os.walk('..\\stylegan2\\results\\00043-generate-images', topdown=True):
    for file in files:
        if file.endswith('.png'):
            index += 1
            if index == batch_size:
                batches.append(current_batch)
                current_batch = []
                index = 0
            
            current_batch.append(root + os.sep + file)

print('making output directories')
if not os.path.exists('output'):
     os.makedirs('output')
for directory in labels:
    if not os.path.exists('output' + os.sep + directory):
        os.makedirs('output' + os.sep + directory)

total = 0
batch_index = 0
for batch in batches:
    print('getting predictions on batch ' + str(batch_index))
    images = []
    for image_file in batch:
        img = image.load_img(image_file, target_size=(image_size, image_size))
        img = image.img_to_array(img)
        img = img.reshape(image_size, image_size, 3)
        images.append(img)
    result = model.predict(np.array(images), batch_size=len(batch))
    total += len(batch)
    batch_index += 1
    print('prediction done on batch ' + str(batch_index) + ', total of ' + str(total) + ' images')
    image_index = 0
    for image_result in result:
        for result_index in range(len(image_result)):
            if image_result[result_index] >= 0.9:
                head, tail = ntpath.split(batch[image_index])
                shutil.move(batch[image_index], os.path.join('output', labels[result_index], tail))
                break
        image_index += 1

    print('finished moving images to category destinations')
print(str(total))
