import os
import sys
import argparse
import shutil
import ntpath
import keras.models as models
from keras.preprocessing import image
import numpy as np

parser = argparse.ArgumentParser(description='predict gan output classes', formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--image-dir', help='directory containing images to classify', required=True)
parser.add_argument('--output-dir', help='directory to write output images to', required=True)
parser.add_argument('--model', help='model to use for predictions', required=True)
args = parser.parse_args()

if not os.path.exists(args.image_dir):
    print ('Error: input images directory does not exist')
    sys.exit(1)
         

labels = ['charcoal-figure-standing', 'charcoal-landscape', 'charcoal-portrait', 'color-portrait-chromatic', 'color-portrait-dark-background', 'field-chromatic-small-horizon', 'landscape-chromatic', 'landscape-with-horizon', 'object-chromatic']

# we're classifying ata from vangogh-256
#image_size = 256
image_size = 224

#model = models.load_model('vangogh-256-classifier-small_last4.h5')
model = models.load_model(args.model)

batch_size = 16
batches = []
current_batch = []
index = -1
for root, dirs, files in os.walk(args.image_dir, topdown=True):
    for file in files:
        if file.endswith('.png'):
            index += 1
            if index == batch_size:
                batches.append(current_batch)
                current_batch = []
                index = 0
            
            current_batch.append(root + os.sep + file)

print('making output directories')
if not os.path.exists(args.output_dir):
     os.makedirs(args.output_dir)
for directory in labels:
    if not os.path.exists(args.output_dir + os.sep + directory):
        os.makedirs(args.output_dir + os.sep + directory)

total = 0
batch_index = 0
for batch in batches:
    print('getting predictions on batch ' + str(batch_index))
    images = []
    for image_file in batch:
        img = image.load_img(image_file, target_size=(image_size, image_size))
        img = image.img_to_array(img)
        img = img.astype('float32')
        img /= 255.0
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
                shutil.move(batch[image_index], os.path.join(args.output_dir, labels[result_index], tail))
                break
        image_index += 1

    print('finished moving images to category destinations')
print(str(total))
