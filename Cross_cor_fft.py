from __future__ import division, unicode_literals, print_function  # for compatibility with Python 2 and 3
import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_float
plt.rcParams['image.cmap'] = 'gray'
from scipy import ndimage
from skimage.feature import register_translation
#from upsampled_dft import register_translation

def fft_cross(image, offset_image, show = False, dark_field = True, GCmode = False):
    """
    calculate cross correlationship map with subpixel resolution
    Parameters:
    ----------
    image: image that want to registered
    offset_image: reference image for registration
    show: if want to plot the image
    dark_field: if also want to calculate the dark field image
    GCmode: measure the center of mass of the cross correlation map
    
    shift: shift vector (in pixels) required to register the two images
    error: translation invariant normalized RMS error 
    diffphase: global phase difference between the two images
    
    """
    # pixel precision first
    shift, error, diffphase = register_translation(image, offset_image, 100) # can change 100 to a larger number if want to detect even smaller pixel shift
    if dark_field:
        # Show the output of a cross-correlation 
        # Caculate dark filed value
        image_product = np.fft.fft2(image) * np.fft.fft2(offset_image).conj()
        cc_image = np.fft.fftshift(np.fft.ifft2(image_product))
        y = int(cc_image.shape[0]/2+shift[0])-1
        x = int(cc_image.shape[1]/2+shift[1])-1
        
        dark_value = cc_image.real[y,x]
    else:
        dark_value = (0, 0)

    if GCmode:
        image_product = np.fft.fft2(image) * np.fft.fft2(offset_image).conj()
        cc_image = np.fft.fftshift(np.fft.ifft2(image_product))
        y, x = ndimage.measurements.center_of_mass(cc_image.real)
        dark_value = cc_image.real[int(y),int(x)]
        shift = (y, x)
        
    if show:
        

        fig = plt.figure(figsize=(8, 3))
        ax1 = plt.subplot(1, 3, 1)
        ax2 = plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1)
        ax3 = plt.subplot(1, 3, 3)

        ax1.imshow(image, cmap='gray')
        ax1.set_axis_off()
        ax1.set_title('Reference image')

        ax2.imshow(offset_image.real, cmap='gray')
        ax2.set_axis_off()
        ax2.set_title('Offset image')

        if dark_field:
            ax3.imshow(cc_image.real)
            ax3.set_axis_off()
            ax3.set_title("Cross-correlation")

        plt.show()

        print("Detected pixel offset (y, x): {}".format(shift))
    cc_image.real

    return(shift, error, diffphase, dark_value)




#caculate displacement
def caculate_displacement (template, np_sand_crop, w, neighour, GCmode = False):
    """
    calculate the pixel shift per pixel to create the differential phase contrast map
    Parameters:
    ----------
    template: image that want to calculate the subpixel shift (image after shift)
    neighour: number of pixel to be included outside the windows
    w: window size
    GCmode: measure the center of mass of the cross correlation map
    
    mylist_x: differential phase contrast image in x direction
    mylist_y: differential phase contrast image in y direction
    mylist_xy: sum of shift vectors
    mylist_intensity: dark field image
    """
    mylist_y=[]
    mylist_x=[]
    mylist_xy=[]
    mylist_intensity=[]
    for i in range (2*w,template.shape[0]-2*w):  # in y direction
        for j in range (2*w,template.shape[1]-2*w): # in x direction
        
            template_roi = np.copy(template[i-w:i+w+1, j-w:j+w+1]) 
            np_sand_crop_roi = np.copy(np_sand_crop[i-w-neighour:i+w+neighour+1, j-w-neighour:j+w+neighour+1]) 
        
            [y, x], _, _ , dark_value=fft_cross(template_roi, np_sand_crop_roi, show = False, dark_field = True, GCmode = GCmode)                          
            #corr = signal.correlate2d(np_sand_crop_roi, template_roi, boundary='symm', mode='same')
                                     
            #y, x = np.unravel_index(np.argmax(corr), corr.shape) # find the match
            #center_y= 2*w
            #center_x= 2*w
            #while abs(y-center_y)>neighour or abs(x-center_x)>neighour:
            #    corr[y,x] = 0
            #    y, x = np.unravel_index(np.argmax(corr), corr.shape) # find the match
                 
            
            mylist_y.append(y) 
            mylist_x.append(x)  
            mylist_intensity.append(dark_value)
            mylist_xy.append(np.sqrt(y**2 + x**2))
    return ({'mylist_x':mylist_x,'mylist_y':mylist_y,'mylist_xy':mylist_xy,'mylist_intensity':mylist_intensity})

# padding zero to the edge
def pad_zero_reshape_image_phase(image_list, reference_image, w, neighour):
        
    """
    Reshape the phase contrast images and pad zero to the output images so that it is the same size as the input image
    Parameters:
    ----------
    image_list: phase contrast image before reshaping, which is a 1D list
    reference_image: raw image 
    w: window size
    neighour: number of pixel to be included outside the windows
    
    image_phase: reshaped image
    """
    image_phase = np.reshape(image_list,[reference_image.shape[0]-4*w,reference_image.shape[1]-4*w])

    new_row = np.zeros((2*w,image_phase.shape[1]))
    image_phase = np.concatenate((image_phase,new_row))
    image_phase = np.concatenate((new_row,image_phase))

    new_col = np.zeros((image_phase.shape[0],2*w))
    image_phase = np.concatenate((image_phase,new_col),axis = 1)
    image_phase = np.concatenate((new_col,image_phase),axis = 1)
    
    return (image_phase)



def imshow_all(*images, **kwargs):
    """ Plot a series of images side-by-side.
    Convert all images to float so that images have a common intensity range.
    Parameters
    ----------
    limits : str
        Control the intensity limits. By default, 'image' is used set the
        min/max intensities to the min/max of all images. Setting `limits` to
        'dtype' can also be used if you want to preserve the image exposure.
    titles : list of str
        Titles for subplots. If the length of titles is less than the number
        of images, empty strings are appended.
    kwargs : dict
        Additional keyword-arguments passed to `imshow`.
    """
    images = [img_as_float(img) for img in images]

    titles = kwargs.pop('titles', [])
    if len(titles) != len(images):
        titles = list(titles) + [''] * (len(images) - len(titles))

    limits = kwargs.pop('limits', 'image')
    if limits == 'image':
        kwargs.setdefault('vmin', min(img.min() for img in images))
        kwargs.setdefault('vmax', max(img.max() for img in images))
    elif limits == 'dtype':
        vmin, vmax = dtype_limits(images[0])
        kwargs.setdefault('vmin', vmin)
        kwargs.setdefault('vmax', vmax)

    nrows, ncols = kwargs.get('shape', (1, len(images)))

    size = nrows * kwargs.pop('size', 5)
    width = size * len(images)
    if nrows > 1:
        width /= nrows * 1.33
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(width, size))
    
    for ax, img, label in zip(axes.ravel(), images, titles):
        ax.imshow(img, **kwargs)
        ax.set_title(label)
        
    plt.show()


    

    
    
    
      


