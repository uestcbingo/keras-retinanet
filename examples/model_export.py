
import  os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorflow as tf
from keras import backend as K
import numpy as np
from matplotlib import pyplot as plt
import time
from PIL import Image
import keras
import sys
sys.path.append('../')
import keras_retinanet
import cv2
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

plt.rcParams['figure.figsize'] = (16,10)
print(tf.__version__)


class DensityModelFreezer():
    def __init__(self,weight_path,save_folder, save_name):
        self.weight_path = weight_path
        self.save_folder = save_folder
        self.save_name = save_name

    def freeze_session(self, session, keep_var_names=None, output_names=None, clear_devices=True):
        from tensorflow.python.framework.graph_util import convert_variables_to_constants
        graph = session.graph
        with graph.as_default():
            freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
            output_names = output_names or []
            output_names += [v.op.name for v in tf.global_variables()]
            input_graph_def = graph.as_graph_def()
            if clear_devices:
                for node in input_graph_def.node:
                    node.device = ""
            frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                          output_names, freeze_var_names,
                                                          variable_names_blacklist=['global_step'])
            return frozen_graph

    def freeze_model(self):
        K.clear_session()
        model1 = models.load_model(self.weight_path)
        model = models.convert_model(model1, nms=True, class_specific_filter=False,
                                     anchor_params=None)
        #model.load_weights(self.weight_path, by_name=True)
        model.summary()
        print([node.op.name for node in model.outputs])
        graph = self.freeze_session(K.get_session(), output_names=[model.output[i].op.name for i in range(len(model.outputs))])
        tf.train.write_graph(graph,self.save_folder, self.save_name,as_text=False)
        print("Done !")


    def load_graph_from_freezer(self):

        with tf.gfile.GFile(self.save_folder +'/'+ self.save_name, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(
                graph_def,
                input_map=None,
                return_elements=None,
                name="prefix",
                op_dict=None,
                producer_op_list=None
            )
            return graph



"""
model.summary()
parallel_model = multi_gpu_model(model, gpus=2)
parallel_model.load_weights(weights_path, by_name=True)
parallel_model.summary()
single_model = parallel_model.layers[-4]
single_model.summary()
single_model.save_weights('/home/crsc/kxb/TrainingModels/density/20190301/single_gpu_model.h5')
print("Done")
"""

def main():
    weight_path = '/home/crsc/kxb/TrainingModels/pedestrian/keras-retina/20190220/resnet50_csv_30.h5'
    save_folder = '/home/crsc/kxb/TrainingModels/pedestrian/keras-retina/pd/'
    save_name = 'retina800_20190326.pb'

    freezer = DensityModelFreezer(
        weight_path= weight_path,save_folder = save_folder,save_name = save_name)
    #freezer.freeze_model()
    graph = freezer.load_graph_from_freezer()

    for op in graph.get_operations():
        print(op.name)
    """
        [u'filtered_detections/map/TensorArrayStack/TensorArrayGatherV3',
         u'filtered_detections/map/TensorArrayStack_1/TensorArrayGatherV3',
         u'filtered_detections/map/TensorArrayStack_2/TensorArrayGatherV3']
    """
    boxes = graph.get_tensor_by_name('prefix/filtered_detections/map/TensorArrayStack/TensorArrayGatherV3:0')
    scores = graph.get_tensor_by_name('prefix/filtered_detections/map/TensorArrayStack_1/TensorArrayGatherV3:0')
    labels = graph.get_tensor_by_name('prefix/filtered_detections/map/TensorArrayStack_2/TensorArrayGatherV3:0')
    x = graph.get_tensor_by_name('prefix/input_1:0')

    img_path = '/home/crsc/kxb/TrainingSamples/Pedestrian/AllSamples-Labelme/20190217/Images/Guangzhou_InStation_Images_1000015.jpg'
    #img_path = '/home/crsc/kxb/TrainingModels/pedestrian/keras-retina/snapshot20170813162017.jpg'
    # load image
    image = read_image_bgr(img_path)

    # copy to draw on
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    # preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image)
    image = np.expand_dims(image,axis=0)
    labels_to_names = {0: 'bg', 1: 'ped', 2: 'rb'}

    # We launch a Session
    with tf.Session(graph=graph) as sess:
        for i in xrange(1, 10, 1):
            begin_t = time.time()
            box_all, score_all, label_all = sess.run([boxes, scores, labels], feed_dict={x: image})
            end_t = time.time()
            print "time used : " + str(end_t - begin_t)
            box_all /= scale

            for box, score, label in zip(box_all[0], score_all[0], label_all[0]):
                if score < 0.2:
                    break
                color = label_color(label)

                b = box.astype(int)
                draw_box(draw, b, color=color)

                caption = "{} {:.3f}".format(labels_to_names[label], score)
                draw_caption(draw, b, caption)

            plt.figure(figsize=(15, 15))
            plt.axis('off')
            plt.imshow(draw)
            plt.show()

if __name__ == '__main__':
    main()

