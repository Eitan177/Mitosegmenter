from skimage.transform import resize
import tensorflow as tf
import os
import sys
import skimage
from skimage.io import imread                                                   
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

json_fileread = open('model.json', 'r')
loaded_model_json = json_fileread.read()
json_fileread.close()

new_model = tf.keras.models.model_from_json(loaded_model_json)
new_model.load_weights('model.fibsem_test_job_0.h5')

image_file = st.file_uploader("Upload one image", type=["png","jpg","jpeg"])
if image_file is not None:
    img2=imread(image_file)
    im2 = Image.fromarray((img2).astype(np.uint8)).convert('L')
    im2.save('oshima.tif')
    img2_resize=resize(imread('oshima.tif'),(768,1024))
    img2_resize=np.expand_dims(img2_resize,axis=2)
    img2_resize=np.expand_dims(img2_resize,axis=0)

    mo=np.zeros(img2_resize.shape)
    for jj in np.arange(1.02,2.1,0.1):
        mo1=new_model.predict(((img2_resize)**(1/jj)))
        mo=np.maximum(mo,mo1)


    fig, ax = plt.subplots(1,2)        
    ax[0].imshow(mo[0,0:,0:,0],cmap='gray')   
    ax[1].imshow(img2_resize[0,:],cmap='gray')
    st.pyplot(fig)

    mm=skimage.measure.find_contours(np.round(mo[0,:,:,0]),0.5,fully_connected='low')
    hh=np.array([len(jj) for jj in mm])
    mito_count=np.sum(hh>100)
    st.write('Mitochondria Count:')
    st.write(mito_count)
    
    def find_area_perim(array):
        a = 0
        p = 0
        ox,oy = array[0]
        for x,y in array[1:]:
            a += (x*oy-y*ox)
            p += abs((x-ox)+(y-oy)*1j)
            ox,oy = x,y
        return a/2,p
    lla=list()
    llp=list()
    llap = list()
    
    mmfilt=np.array(mm,dtype='object')[hh>100]
    for jj in mmfilt:
        a,p=find_area_perim(jj)
        lla.append(a)
        llp.append(p)
        llap.append(p/a)
    fig, ax = plt.subplots()
    ax.hist(llp, bins=20)
    st.pyplot(fig)
    st.write('mean perimeter '+str(np.mean(llp)))
    st.write('median perimeter '+str(np.median(llp)))
    st.write('min perimeter '+str(np.min(llp)))
    st.write('max perimeter '+str(np.max(llp)))
