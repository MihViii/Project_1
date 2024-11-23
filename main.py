# 2.4
import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import pytesseract as pt
import plotly.express as px
import matplotlib.pyplot as plt
import xml.etree.ElementTree as xet

from glob import glob
from skimage import io
from shutil import copy
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

    # trich anh ra
import glob

#     # thư mục chứa file DataSet
# base_dir = os.path.dirname(__file__)

    # Sử dụng glob để tìm các tệp .xml trong đường dẫn cục bộ trên máy tính
# path = glob.glob(os.path.join(base_dir, 'DataSet/images/*.xml'))

path = glob.glob(r'./DataSet/images/*.xml')

    # Tạo dictionary
labels_dict = dict(filepath=[],xmin=[],xmax=[],ymin=[],ymax=[])

    # Duyệt các file xml trong đường dẫn
for i in path:
    info = xet.parse(i)
    root = info.getroot()
    member_object = root.find('object')
    labels_info = member_object.find('bndbox')
    xmin = int(labels_info.find('xmin').text)
    xmax = int(labels_info.find('xmax').text)
    ymin = int(labels_info.find('ymin').text)
    ymax = int(labels_info.find('ymax').text)

    labels_dict['filepath'].append(i)
    labels_dict['xmin'].append(xmin)
    labels_dict['xmax'].append(xmax)
    labels_dict['ymin'].append(ymin)
    labels_dict['ymax'].append(ymax)

df = pd.DataFrame(labels_dict)
df.to_csv('labels.csv',index=False)
print(df.head())

   # Hàm lấy đường dẫn ảnh
filename = df['filepath'][0]
def getFilename(filename):
    filename_image = xet.parse(filename).getroot().find('filename').text
    # filepath_image = os.path.join('DataSet/images',filename_image)
    filepath_image = os.path.join('./DataSet/images', filename_image)
    return filepath_image
getFilename(filename)

image_path = list(df['filepath'].apply(getFilename))
print(image_path[:10])   # random check

# 2.5 xác thực dữ liệu
file_path = image_path[0]   # dẫn tới ảnh đầu tiên
img = cv2.imread(file_path) # read the image
img = io.imread(file_path) # Read the image
fig = px.imshow(img)
fig.update_layout(width=600, height=500, margin=dict(l=10, r=10, b=10, t=10),xaxis_title='Figure 8')
fig.add_shape(type='rect',x0=144, x1=427, y0=119, y1=318, xref='x', yref='y',line_color='cyan')
fig.show() # thêm để mở ảnh trong trình duyệt
labels_dict_df = pd.DataFrame.from_dict(labels_dict)
labels_dict_df.iloc[ : , 1 : ].values

#Targeting all our values in array selecting all columns
labels = df.iloc[ : , 1 : ].values
data = []
output = []
for ind in range(len(image_path)):
    image = image_path[ind]
    img_arr = cv2.imread(image)
    if img_arr is None:
        continue
    else:
        h,w,d = img_arr.shape
    # Prepprocesing
    load_image = load_img(image,target_size=(224,224))
    load_image_arr = img_to_array(load_image)
    norm_load_image_arr = load_image_arr/255.0 # Normalization
    # Normalization to labels
    xmin,xmax,ymin,ymax = labels[ind]
    nxmin,nxmax = xmin/w,xmax/w
    nymin,nymax = ymin/h,ymax/h
    label_norm = (nxmin,nxmax,nymin,nymax) # Normalized output
    # Append
    data.append(norm_load_image_arr)
    output.append(label_norm)


# Convert data to array
X = np.array(data,dtype=np.float32)
y = np.array(output,dtype=np.float32)

# Split the data into training and testing set using sklearn.
x_train,x_test,y_train,y_test = train_test_split(X,y,train_size=0.8,random_state=0)
#x_train.shape,x_test.shape,y_train.shape,y_test.shape

inception_resnet = InceptionResNetV2(weights="imagenet",include_top=False, input_tensor=Input(shape=(224,224,3)))
# ---------------------
headmodel = inception_resnet.output
headmodel = Flatten()(headmodel)
headmodel = Dense(500,activation="relu")(headmodel)
headmodel = Dense(250,activation="relu")(headmodel)
headmodel = Dense(4,activation='sigmoid')(headmodel)


# ---------- model
model = Model(inputs=inception_resnet.input,outputs=headmodel)

# Complie model
model.compile(loss='mse',optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))
model.summary()

tfb = TensorBoard('object_detection')

history = model.fit(x=x_train,y=y_train,batch_size=10,epochs=100,validation_data=(x_test,y_test),callbacks=[tfb])
model.save('./my_model.keras')
