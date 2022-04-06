#install method

#cpu https://cv.gluon.ai/install/install-more.html

"""
if on amd
conda install pyqt==5.9.2 opencv-python==4.3.0.36 
maybe pip install opencv-python==4.3.0.36 
"""


#import torch




#print(torch.cuda.is_available())
from gluoncv import model_zoo, data, utils
from matplotlib import pyplot as plt


net = model_zoo.get_model('yolo3_darknet53_voc', pretrained=True)

#im_fname = utils.download('https://raw.githubusercontent.com/zhreshold/' +
#                          'mxnet-ssd/master/data/demo/dog.jpg',
#                          path='dog.jpg')

file_name = 'car.jpg'

x, img = data.transforms.presets.yolo.load_test(file_name, short=512)
print('Shape of pre-processed image:', x.shape)

class_IDs, scores, bounding_boxs = net(x)

ax = utils.viz.plot_bbox(img, bounding_boxs[0], scores[0],
                         class_IDs[0], class_names=net.classes)
plt.show()