
import keras.callbacks as cbks
import tensorflow as tf
import csv
import random
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import io
plt.rcParams['figure.figsize']=(20,15)
import os

sys.path.append('../../')
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
#import models
#from utils.image import read_image_bgr, preprocess_image, resize_image
#from utils.visualization import draw_box, draw_caption
#from utils.colors import label_color


test_list =[]
test_file_path = '/home/crsc/kxb/TrainingSamples/Pedestrian/AllSamples-Labelme/20190217/test_list.csv'
with open(test_file_path, 'r') as f:
    csv_reader = csv.reader(f)
    for row in csv_reader:
        test_list.append(row[0])

print(len(test_list))

def predict(model):
    id = np.random.randint(low=0,high=len(test_list))
    label_name = test_list[id]
    image_name = ''
    if(os.path.splitext(label_name)[1] == '.txt'):
        image_name = label_name.replace('Label','Image').replace('.txt','.jpg')
    else:
        image_name = label_name.replace('Labels', 'Images').replace('.json', '.jpg')
    labels_to_names = {0: 'bg', 1: 'ped', 2: 'rb'}
    # load image
    image = read_image_bgr(image_name)

    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    image = preprocess_image(image)
    image, scale = resize_image(image)
    model = models.convert_model(model, nms=True, class_specific_filter=False,
                                 anchor_params=None)
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    # correct for image scale
    boxes /= scale


    # visualize detections
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        if score < 0.5:
            break

        color = label_color(label)

        b = box.astype(int)
        draw_box(draw, b, color=color)

        caption = "{} {:.3f}".format(labels_to_names[label], score)
        draw_caption(draw, b, caption)

    #plt.figure(figsize=(15, 15))
    plt.axis('off')
    plt.imshow(draw)
    #plt.show()

    output = io.BytesIO()
    plt.savefig(output, format='png')
    image_string = output.getvalue()
    output.close()
    plt.close()
    img_width, img_height = draw.shape[0], draw.shape[1]
    tf_im =tf.Summary.Image(height=img_height,
                     width=img_width,
                     colorspace=3,
                     encoded_image_string=image_string)
    return tf_im


class VisualizeTrainingResult(cbks.Callback):
    def __init__(self,writer):
        super(VisualizeTrainingResult,self).__init__()
        self.writer = tf.summary.FileWriter(writer)


    def on_batch_end(self, batch, logs=None):
        if (batch % 500 == 0):
            image = predict( self.model)
            summary = tf.Summary(value=[tf.Summary.Value(tag='predict', image=image)])
            #writer = tf.summary.FileWriter('/home/crsc/kxb/TrainingLog/Pedestrain/keras-retina/20190221/')
            self.writer.add_summary(summary,batch)


    """
    def on_epoch_end(self, epoch, logs={}):

        if(epoch %100 == 0):
            image = invasion_inference(image_lists, self.model)
            summary = tf.Summary(value=[tf.Summary.Value(tag='invasion', image=image)])
            writer = tf.summary.FileWriter('/home/crsc/kxb/TrainingSamples/CrossLine/log/')
            writer.add_summary(summary, epoch)
            writer.close()
        if(epoch % 100 == 0):
            #save models
            model_folder = '/home/crsc/kxb/TrainingSamples/CrossLine/models/'
            self.model.save(model_folder + "invasion_model.h5")
            self.model.save_weights(model_folder + "invasion_weights.h5")
        """