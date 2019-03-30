import hdf5storage
import matplotlib.pyplot as plt
from Cross_cor_fft import caculate_displacement, pad_zero_reshape_image_phase, imshow_all
import numpy as np
import scipy.io as sio
path = .. # image folder
filename = .. # name of the input images

data = hdf5storage.loadmat(path+'\\'+filename, squeeze_me=True)
#%%
test_2D = data['I'] # speckle image
shift_2D = data['Is'] # sample + speckle
Ori_2D = data['Io'] # absorption image

show = True

if show:
    fig = plt.figure(figsize=(5,15))
    plt.subplot(311)
    plt.imshow(test_2D) #check if it shift
    plt.subplot(312)
    plt.imshow(shift_2D) #check if it shift
    plt.subplot(313)
    plt.imshow(np.divide(test_2D, Ori_2D)) #check if it shift
    plt.show()


#%%
w = 8
n = 1
#np_sand_crop = np.divide(test_2D, Ori_2D)
np_sand_crop = test_2D
results_w = caculate_displacement (template = shift_2D,np_sand_crop = np_sand_crop,w = w,neighour = 0)
sio.savemat('Phase_img', results_w, appendmat=True, format='5', long_field_names=False, do_compression=False, oned_as='row')
    
image_phase_x = pad_zero_reshape_image_phase(results_w['mylist_x'], shift_2D, w, neighour = n)
image_phase_y = pad_zero_reshape_image_phase(results_w['mylist_y'], shift_2D, w, neighour = n)

imshow_all(image_phase_x,image_phase_y)
