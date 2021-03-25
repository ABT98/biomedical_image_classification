#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ata
"""
"""Gerekli kütüphaneler"""
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam

# resimlerimizin boyutu
goruntu_genislik, goruntu_yukseklik = 299, 299

#eğitim verisinin lokasyonu
egitim_seti_dizin = 'veriseti/egitim' 
#doğrulama(validation) verisinin lokasyonu
dogrulama_seti_dizin = 'veriseti/dogrulama' 

# samples_per_epoch'u belirlemek için kullanılan örnek sayısı
nb_train_samples = 65
nb_validation_samples = 10

# eğitim verisinden geçiş sayısı (tur)
epochs = 20
#aynı anda işlenen görüntü sayısı
batch_size = 5  

""" Veri ön işleme Data Augmentation  """
egitim_veri_uretici = ImageDataGenerator(
        rescale=1./255,            # piksel değerlerini [0,1] normalize eder
        shear_range=0.2,      
        zoom_range=0.2,    
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)  


dogrulama_veri_uretici = ImageDataGenerator(
         rescale=1./255)      # piksel değerlerini [0,1] normalize eder


train_generator = egitim_veri_uretici.flow_from_directory(
    egitim_seti_dizin,
    target_size=(goruntu_yukseklik, goruntu_genislik),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = dogrulama_veri_uretici.flow_from_directory(
    dogrulama_seti_dizin,
    target_size=(goruntu_yukseklik, goruntu_genislik),
    batch_size=batch_size,
    class_mode='binary')

"""Orijinal Inception V3 modeliyle başlayın. Ardından, üst veya tamamen bağlı
 katmanları orijinal ağdan kaldırın. ImageNet'ten önceden alınmış ağırlıkları 
 kullanın"""

""" Inception modelini tanıyalım """
from keras.applications.inception_v3 import InceptionV3

deneme_model= InceptionV3()

print(deneme_model.summary())

 
temel_model = applications.InceptionV3(weights='imagenet', include_top=False, input_shape=(goruntu_genislik, goruntu_yukseklik, 3))


""" Orijinal modelin üstüne yeni katmanlar ekliyoruz. 
Birçok olasılık var, ancak burada küresel bir ortalama havuz katmanı, 
256 düğümlü tamamen bağlı bir katman, dropout ve sigmoid aktivasyonu ekliyoruz. 
Ayrıca bir optimize edici tanımladık; Adam optimizer
"""

ust_model = Sequential()
ust_model.add(GlobalAveragePooling2D(input_shape=temel_model.output_shape[1:], data_format=None)),  
ust_model.add(Dense(256, activation='relu'))
ust_model.add(Dropout(0.5))
ust_model.add(Dense(1, activation='sigmoid')) 

model = Model(inputs=temel_model.input, outputs=ust_model(temel_model.output))

model.compile(optimizer=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08,decay=0.0), loss='binary_crossentropy', metrics=['accuracy'])


""" Daha sonra, modeli, modeli çalıştırmak(eğitmek ve doğrulamak) için son kod kümesi olan generator a fit ediyoruz. """

gecmis = model.fit_generator(
            train_generator,
            steps_per_epoch=nb_train_samples // batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=nb_validation_samples // batch_size)

"""Grafik çizme  """

import matplotlib.pyplot as plt

print(gecmis.history.keys())

plt.figure()
plt.plot(gecmis.history['acc'], 'orange', label='Egitim Dogruluk')
plt.plot(gecmis.history['val_acc'], 'blue', label='Dogrulama Dogruluk')
plt.plot(gecmis.history['loss'], 'red', label='Egitim Kayıp')
plt.plot(gecmis.history['val_loss'], 'green', label='Dogrulama Kayıp')
plt.legend()
plt.show()


""" Test verisini kullanarak modele tahmin yaptırma"""

import numpy as np
from keras.preprocessing import image

goruntu_dizin='veriseti/test/gogus.png' 
goruntu_dizin2='veriseti/test/karin.png'  
goruntu = image.load_img(goruntu_dizin, target_size=(goruntu_genislik, goruntu_yukseklik))
goruntu2 = image.load_img(goruntu_dizin2, target_size=(goruntu_genislik, goruntu_yukseklik))

plt.imshow(goruntu)
plt.show()

goruntu= image.img_to_array(goruntu)
x = np.expand_dims(goruntu, axis=0) * 1./255
score = model.predict(x)
print('Tahmin:', score, 'Göğüs Röntgen' if score < 0.5 else 'Karın Röntgen')

plt.imshow(goruntu2)
plt.show()

goruntu = image.img_to_array(goruntu2)
x = np.expand_dims(goruntu2, axis=0) * 1./255
score2 = model.predict(x)
print('Tahmin:', score2, 'Göğüs Röntgen' if score2 < 0.5 else 'Karın Röntgen ')

""" Modeli kayıt etme"""
from keras.models import load_model

model.save('my_model.h5')  #   'my_model.h5' isminde bir HDF5 dosyası oluşturur.

model_yeni = load_model('my_model.h5')
print(model_yeni.summary())

""" Modelin ağırlılarını kayıt etme"""
model.save_weights('my_model_weights.h5')

model.load_weights('my_model_weights.h5')

""" Ayrıntılı bilgi için
https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model 
"""