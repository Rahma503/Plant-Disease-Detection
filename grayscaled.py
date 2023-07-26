
pip install tensorflow


# In[4]:


import tensorflow as tf
from tensorflow.keras import models,layers
import matplotlib.pyplot as plt


# In[4]:


data = tf.keras.preprocessing.image_dataset_from_directory("D:\code\plantvillage dataset\grayscale\Potato" ,shuffle=True ,image_size=(256,256),batch_size=32)


# In[5]:


for img_batch , lable_batch in data.take(1):
    plt.imshow(img_batch[0].numpy().astype("uint8"))
    plt.axis("off")


# In[6]:


for img_batch , lable_batch in data.take(1):
   print(img_batch.shape)


# In[7]:


def get_dataset_partitions_tf(ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=1000):
    ds_size = len(ds)
    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=12)
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)
    return train_ds, val_ds, test_ds


# In[8]:


train_ds,val_ds,test_ds =get_dataset_partitions_tf(data)


# In[9]:


len(train_ds)


# In[10]:


train_ds, val_ds, test_ds = get_dataset_partitions_tf(data)
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)


# In[11]:


resize_rescale = tf.keras.Sequential([
    layers.experimental.preprocessing.Resizing(256, 256),
    layers.experimental.preprocessing.Rescaling(1.0/255)
])

data_augman = tf.keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    layers.experimental.preprocessing.RandomRotation(0.2)
])


# In[12]:


input_shape = (32,256, 256, 3)
model = models.Sequential([
    resize_rescale, data_augman, layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),
    resize_rescale, data_augman, layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    resize_rescale, data_augman, layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    resize_rescale, data_augman, layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    resize_rescale, data_augman, layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    resize_rescale, data_augman, layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(3, activation='softmax')
])
model.build(input_shape=input_shape)


# In[13]:


model.compile(
optimizer='adam',
loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)   ,
metrics=['accuracy']    
)


# In[14]:


his= model.fit(train_ds,epochs=5,batch_size=32,verbose=1,validation_data=val_ds)


# In[15]:


scores= model.evaluate(test_ds)
import numpy as np


# In[18]:


class_names_pot= data.class_names
class_names_pot


# In[24]:


def predict(model, img,class_names):
    img_arr = tf.image.resize(img, (224, 224))  # Resize the image to match the input shape of the model
    img_arr = tf.keras.applications.resnet50.preprocess_input(img_arr)  # Preprocess the image
    
    img_arr = tf.expand_dims(img_arr, 0)  # Add an extra dimension to match the expected input shape of the model
    
    predictions = model.predict(img_arr)  # Predict the class probabilities
    
    prediction_class = class_names[np.argmax(predictions)]  # Get the predicted class
    confidence = round(100 * np.max(predictions), 2)  # Get the confidence of the prediction
    
    return prediction_class, confidence


# In[25]:


for images, labels in test_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        prediction_class, confidence = predict(model, images[i].numpy(),class_names_pot)
        actual_class = class_names_pot[labels[i]]
        plt.title(f"Actual:{actual_class},\n Prediction : {prediction_class},\n confidence {confidence}")
        plt.axis("off")


# In[27]:


data_ch = tf.keras.preprocessing.image_dataset_from_directory("D:\code\plantvillage dataset\grayscale\Cherry" ,shuffle=True ,image_size=(256,256),batch_size=32)


# In[28]:


train_ds_ch,val_ds_ch,test_ds_ch =get_dataset_partitions_tf(data_ch)


# In[29]:


len(train_ds_ch)


# In[30]:


train_ds_ch, val_ds_ch, test_ds_ch = get_dataset_partitions_tf(data_ch)
train_ds_ch = train_ds_ch.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds_ch = val_ds_ch.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds_ch = test_ds_ch.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

input_shape = (32,256, 256, 3)
model_ch = models.Sequential([
    resize_rescale, data_augman, layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),
    resize_rescale, data_augman, layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    resize_rescale, data_augman, layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    resize_rescale, data_augman, layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    resize_rescale, data_augman, layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    resize_rescale, data_augman, layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(2, activation='softmax')
])
model_ch.build(input_shape=input_shape)


model_ch.compile(
optimizer='adam',
loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)   ,
metrics=['accuracy']    
)


# In[31]:


his_ch= model_ch.fit(train_ds_ch,epochs=5,batch_size=32,verbose=1,validation_data=val_ds_ch)


# In[32]:


scores_ch= model_ch.evaluate(test_ds_ch)


# In[33]:


class_names_ch= data_ch.class_names
class_names_ch


# In[34]:


for images, labels in test_ds_ch.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        prediction_class, confidence = predict(model_ch, images[i].numpy(),class_names_ch)
        actual_class = class_names_ch[labels[i]]
        plt.title(f"Actual:{actual_class},\n Prediction : {prediction_class},\n confidence {confidence}")
        plt.axis("off")


# In[35]:


data_str = tf.keras.preprocessing.image_dataset_from_directory("D:\code\plantvillage dataset\grayscale\strawberry" ,shuffle=True ,image_size=(256,256),batch_size=32)


# In[36]:


train_ds_str,val_ds_str,test_ds_str =get_dataset_partitions_tf(data_str)


# In[37]:


len(train_ds_str)


# In[38]:


train_ds_str = train_ds_str.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds_str = val_ds_str.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds_str = test_ds_str.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

input_shape = (32,256, 256, 3)
model_str = models.Sequential([
    resize_rescale, data_augman, layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),
    resize_rescale, data_augman, layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    resize_rescale, data_augman, layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    resize_rescale, data_augman, layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    resize_rescale, data_augman, layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    resize_rescale, data_augman, layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(2, activation='softmax')
])
model_str.build(input_shape=input_shape)


model_str.compile(
optimizer='adam',
loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)   ,
metrics=['accuracy']    
)


# In[39]:


his_str= model_str.fit(train_ds_str,epochs=5,batch_size=32,verbose=1,validation_data=val_ds_str)


# In[40]:


scores_str= model_str.evaluate(test_ds_str)
class_names_str= data_str.class_names
class_names_str

for images, labels in test_ds_str.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        prediction_class, confidence = predict(model_str, images[i].numpy(),class_names_str)
        actual_class = class_names_str[labels[i]]
        plt.title(f"Actual:{actual_class},\n Prediction : {prediction_class},\n confidence {confidence}")
        plt.axis("off")


# In[41]:


data_pep = tf.keras.preprocessing.image_dataset_from_directory("D:\code\plantvillage dataset\grayscale\Pepper" ,shuffle=True ,image_size=(256,256),batch_size=32)


# In[42]:


train_ds_pep,val_ds_pep,test_ds_pep=get_dataset_partitions_tf(data_pep)
len(train_ds_pep)


# In[43]:


train_ds_pep = train_ds_pep.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds_pep = val_ds_pep.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds_pep = test_ds_pep.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

input_shape = (32,256, 256, 3)
model_pep = models.Sequential([
    resize_rescale, data_augman, layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),
    resize_rescale, data_augman, layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    resize_rescale, data_augman, layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    resize_rescale, data_augman, layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    resize_rescale, data_augman, layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    resize_rescale, data_augman, layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(2, activation='softmax')
])
model_pep.build(input_shape=input_shape)


model_pep.compile(
optimizer='adam',
loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)   ,
metrics=['accuracy']    
)


# In[44]:


his_pep= model_pep.fit(train_ds_pep,epochs=5,batch_size=32,verbose=1,validation_data=val_ds_pep)


# In[45]:


scores_pep= model_pep.evaluate(test_ds_pep)
class_names_pep= data_pep.class_names
class_names_pep

for images, labels in test_ds_pep.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        prediction_class, confidence = predict(model_pep, images[i].numpy(),class_names_pep)
        actual_class = class_names_pep[labels[i]]
        plt.title(f"Actual:{actual_class},\n Prediction : {prediction_class},\n confidence {confidence}")
        plt.axis("off")


# In[46]:


data_p = tf.keras.preprocessing.image_dataset_from_directory("D:\code\plantvillage dataset\grayscale\Peach" ,shuffle=True ,image_size=(256,256),batch_size=32)


# In[47]:


train_ds_p,val_ds_p,test_ds_p=get_dataset_partitions_tf(data_p)
len(train_ds_p)


# In[48]:


train_ds_p = train_ds_p.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds_p = val_ds_p.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds_p = test_ds_p.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

input_shape = (32,256, 256, 3)
model_p = models.Sequential([
    resize_rescale, data_augman, layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),
    resize_rescale, data_augman, layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    resize_rescale, data_augman, layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    resize_rescale, data_augman, layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    resize_rescale, data_augman, layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    resize_rescale, data_augman, layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(2, activation='softmax')
])
model_p.build(input_shape=input_shape)


model_p.compile(
optimizer='adam',
loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)   ,
metrics=['accuracy']    
)


# In[49]:


his_p= model_p.fit(train_ds_p,epochs=5,batch_size=32,verbose=1,validation_data=val_ds_p)


# In[50]:


scores_p= model_p.evaluate(test_ds_p)
class_names_p= data_p.class_names
class_names_p

for images, labels in test_ds_p.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        prediction_class, confidence = predict(model_p, images[i].numpy(),class_names_p)
        actual_class = class_names_p[labels[i]]
        plt.title(f"Actual:{actual_class},\n Prediction : {prediction_class},\n confidence {confidence}")
        plt.axis("off")


# In[52]:


data_A= tf.keras.preprocessing.image_dataset_from_directory("D:\code\plantvillage dataset\grayscale\Apple" ,shuffle=True ,image_size=(256,256),batch_size=32)


# In[5]:


train_ds_a,val_ds_a,test_ds_a=get_dataset_partitions_tf(data_A)
len(train_ds_a)


# In[48]:


train_ds_a = train_ds_a.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds_a = val_ds_a.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds_a = test_ds_a.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

input_shape = (32,256, 256, 3)
model_a = models.Sequential([
    resize_rescale, data_augman, layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),
    resize_rescale, data_augman, layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    resize_rescale, data_augman, layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    resize_rescale, data_augman, layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    resize_rescale, data_augman, layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    resize_rescale, data_augman, layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(4, activation='softmax')
])
model_a.build(input_shape=input_shape)


model_a.compile(
optimizer='adam',
loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)   ,
metrics=['accuracy']    
)


# In[6]:


his_a= model_a.fit(train_ds_a,epochs=5,batch_size=32,verbose=1,validation_data=val_ds_a)


# In[7]:


scores_a= model_a.evaluate(test_ds_a)
class_names_a= data_A.class_names
class_names_a

for images, labels in test_ds_a.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        prediction_class, confidence = predict(model_a, images[i].numpy(),class_names_a)
        actual_class = class_names_a[labels[i]]
        plt.title(f"Actual:{actual_class},\n Prediction : {prediction_class},\n confidence {confidence}")
        plt.axis("off")


# In[8]:


data_t= tf.keras.preprocessing.image_dataset_from_directory("D:\code\plantvillage dataset\grayscale\Tomato" ,shuffle=True ,image_size=(256,256),batch_size=32)


# In[9]:


train_ds_t,val_ds_t,test_ds_t=get_dataset_partitions_tf(data_t)
len(train_ds_t)


# In[48]:


train_ds_t = train_ds_t.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds_t = val_ds_t.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds_t = test_ds_t.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

input_shape = (32,256, 256, 3)
model_t = models.Sequential([
    resize_rescale, data_augman, layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),
    resize_rescale, data_augman, layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    resize_rescale, data_augman, layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    resize_rescale, data_augman, layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    resize_rescale, data_augman, layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    resize_rescale, data_augman, layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
model_t.build(input_shape=input_shape)


model_t.compile(
optimizer='adam',
loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)   ,
metrics=['accuracy']    
)


# In[10]:


his_t= model_t.fit(train_ds_t,epochs=5,batch_size=32,verbose=1,validation_data=val_ds_t)


# In[11]:


scores_t= model_t.evaluate(test_ds_t)
class_names_t= data_t.class_names
class_names_t

for images, labels in test_ds_t.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        prediction_class, confidence = predict(model_t, images[i].numpy(),class_names_t)
        actual_class = class_names_t[labels[i]]
        plt.title(f"Actual:{actual_class},\n Prediction : {prediction_class},\n confidence {confidence}")
        plt.axis("off")


# In[12]:


data_g= tf.keras.preprocessing.image_dataset_from_directory("D:\code\plantvillage dataset\grayscale\Grape" ,shuffle=True ,image_size=(256,256),batch_size=32)


# In[14]:


train_ds_g,val_ds_g,test_ds_g=get_dataset_partitions_tf(data_g)
len(train_ds_g)


# In[48]:


train_ds_g = train_ds_g.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds_g = val_ds_g.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds_g = test_ds_g.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

input_shape = (32,256, 256, 3)
model_g = models.Sequential([
    resize_rescale, data_augman, layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),
    resize_rescale, data_augman, layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    resize_rescale, data_augman, layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    resize_rescale, data_augman, layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    resize_rescale, data_augman, layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    resize_rescale, data_augman, layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(4, activation='softmax')
])
model_g.build(input_shape=input_shape)


model_g.compile(
optimizer='adam',
loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)   ,
metrics=['accuracy']    
)


# In[15]:


his_g= model_g.fit(train_ds_g,epochs=5,batch_size=32,verbose=1,validation_data=val_ds_g)


# In[16]:


scores_g= model_g.evaluate(test_ds_g)
class_names_g= data_g.class_names
class_names_g

for images, labels in test_ds_g.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        prediction_class, confidence = predict(model_g, images[i].numpy(),class_names_g)
        actual_class = class_names_g[labels[i]]
        plt.title(f"Actual:{actual_class},\n Prediction : {prediction_class},\n confidence {confidence}")
        plt.axis("off")


# In[17]:


data_c= tf.keras.preprocessing.image_dataset_from_directory("D:\code\plantvillage dataset\grayscale\Corn" ,shuffle=True ,image_size=(256,256),batch_size=32)


# In[18]:


train_ds_c,val_ds_c,test_ds_c=get_dataset_partitions_tf(data_c)
len(train_ds_c)


# In[48]:


train_ds_c = train_ds_c.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds_c = val_ds_c.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds_c = test_ds_c.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

input_shape = (32,256, 256, 3)
model_c = models.Sequential([
    resize_rescale, data_augman, layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),
    resize_rescale, data_augman, layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    resize_rescale, data_augman, layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    resize_rescale, data_augman, layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    resize_rescale, data_augman, layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    resize_rescale, data_augman, layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(4, activation='softmax')
])
model_c.build(input_shape=input_shape)


model_c.compile(
optimizer='adam',
loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)   ,
metrics=['accuracy']    
)


# In[19]:


his_c= model_c.fit(train_ds_c,epochs=5,batch_size=32,verbose=1,validation_data=val_ds_c)


# In[20]:


scores_c= model_c.evaluate(test_ds_c)
class_names_c= data_c.class_names
class_names_c

for images, labels in test_ds_c.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        prediction_class, confidence = predict(model_c, images[i].numpy(),class_names_c)
        actual_class = class_names_c[labels[i]]
        plt.title(f"Actual:{actual_class},\n Prediction : {prediction_class},\n confidence {confidence}")
        plt.axis("off")
