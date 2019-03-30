'''
functions to caculate the minimum of cost function
Known
Reference image (sandpaper only): Img_ref
Sample image(sandpaper and sample): Img_sample

Img_ref = Io + Ir(x,y)
Img_sample = T*[Io + D*Ir(x - dx, y - dy)]

Minimize function
[Img_sample (recorded) - Img_sample(caculated)]^2 + gradient(T(x,y)) + gradient D(x,y) + dx L2 norm + dy L2 norm
with constrain that -1<dx<1 and -1<dy<1
'''
import numpy as np
from scipy.ndimage import fourier_shift
import hdf5storage
import matplotlib.pyplot as plt
 
from scipy.optimize import minimize
from scipy.signal import convolve2d
import math
import scipy.io as spio







def createCircularMask(h, w, center=None, radius=None):
    #draw a circular mask
    if center is None: # use the middle of the image
        center = [int(w/2), int(h/2)]
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask


def cir_tophat(r):
    n_pixel = int(2*r+1)
    img_tophat = np.ones((n_pixel,n_pixel))
    #print('number of pixel:', np.shape(img_tophat))
    mask = createCircularMask(n_pixel, n_pixel)
    #print('mask size:', np.shape(mask))
    img_tophat = img_tophat*mask

    return(img_tophat)

def create_window(r):
    Tophat1 = cir_tophat(pixel_n)
    Tophat2 = cir_tophat(pixel_n/2)
    window1 = convolve2d(Tophat1, Tophat2)
    window = window1/np.sum(window1)
    return(window)

def our_cost(X):
    #T = X[0]
    #D = X[1]
    dx = X[0]
    dy = X[1]
    T_expect = I_xy_roi/(Io+d_Ir_xy_roi)
            # print('expecting T', np.sum(T_expect*window))
            # shift d_Ir_xy_roi by dx and dy
    if mode_TD == 1:
        temp = fourier_shift(np.fft.fftn(d_Ir_xy_roi), (dy, dx)) # shift the reference image of sandpaper
        Ir_xy_shift = np.fft.ifftn(temp)
        Ir_xy_shift = Ir_xy_shift.real # delta_Ir(x-dx, y-dy)
        #Ir_xy_shift = d_Ir_xy_roi
            

        assert np.shape(window) == np.shape(I_xy_roi), " size doesn't match!"
        Img_avg_roi = window*I_xy_roi
        I_avg_roi = np.sum(window*I_xy_roi)# I_mean
            
            
        assert np.shape(Ir_xy_shift) == np.shape(window), " size doesn't match!"

        T_upper_left =I_avg_roi* np.sum((window*(Ir_xy_shift*Ir_xy_shift)))
        T_upper_right1 = np.sum(window*Ir_xy_shift)
        T_upper_right2 = np.sum(window*I_xy_roi*Ir_xy_shift)
        T_upper_right=T_upper_right1*T_upper_right2
        T_bottom_left = np.sum(window*Ir_xy_shift*Ir_xy_shift)
        T_bottom_right = (np.sum(window*Ir_xy_shift))**2
            

        T = T_upper_left - T_upper_right       
        T = T/(T_bottom_left - T_bottom_right)
        T = T/Io
        #print('calculated T:',T)

        D_upper = np.sum((window*I_xy_roi)*Ir_xy_shift)- I_avg_roi*(np.sum(window*Ir_xy_shift))
        D_lower = T*(np.sum(window*(Ir_xy_shift*Ir_xy_shift))-(np.sum(window*Ir_xy_shift))**2)
    
        D = D_upper/D_lower
        #print('D calculated:', D)

        I_model = T*(Io+D*Ir_xy_shift)
        cost = np.sum(((I_model-I_xy_roi)**2)*window)
        #print('cost:', cost)
   

        return cost

def caculate_TD(d_Ir_xy_roi, I_xy_roi, window, dy, dx):
    temp = fourier_shift(np.fft.fftn(d_Ir_xy_roi), (dy, dx)) # shift the reference image of sandpaper
    Ir_xy_shift = np.fft.ifftn(temp)
    Ir_xy_shift = Ir_xy_shift.real # delta_Ir(x-dx, y-dy)
    #Ir_xy_shift = d_Ir_xy_roi
            

    assert np.shape(window) == np.shape(I_xy_roi), " size doesn't match!"
    Img_avg_roi = window*I_xy_roi
    I_avg_roi = np.sum(window*I_xy_roi)# I_mean
            
            
    assert np.shape(Ir_xy_shift) == np.shape(window), " size doesn't match!"

    T_upper_left =I_avg_roi* np.sum((window*(Ir_xy_shift*Ir_xy_shift)))
    T_upper_right1 = np.sum(window*Ir_xy_shift)
    T_upper_right2 = np.sum(window*I_xy_roi*Ir_xy_shift)
    T_upper_right = T_upper_right1*T_upper_right2
    T_bottom_left = np.sum(window*Ir_xy_shift*Ir_xy_shift)
    T_bottom_right = (np.sum(window*Ir_xy_shift))**2
            

    T = T_upper_left - T_upper_right       
    T = T/(T_bottom_left - T_bottom_right)
    T = T/Io
    #print('calculated T:',T)

    D_upper = np.sum((window*I_xy_roi)*Ir_xy_shift)- I_avg_roi*(np.sum(window*Ir_xy_shift))
    D_lower = T*(np.sum(window*(Ir_xy_shift*Ir_xy_shift))-(np.sum(window*Ir_xy_shift))**2)
    
    D = D_upper/D_lower

    return(T, D)

def caculate_displacement (I_xy, Ir_xy, w):
    global I_xy_roi
    global d_Ir_xy_roi
    global Io
    global mode_TD

 

    mode_TD = 1
    
    Io = np.mean(Ir_xy)
    d_Ir_xy = Ir_xy - Io
    
    mylist_T=[]
    mylist_D=[]
    mylist_dx=[]
    mylist_dy=[]

    
    for i in range (w,img_shape[0]-w):  # in y direction
        for j in range (w,img_shape[1]-w): # in x direction
    
            
        
            # sandpaper + sample in ROI which match the window size
            I_xy_roi = I_xy[i-w:i+w+1, j-w:j+w+1] # I
           
            
            # mean centered in zero's sandpaper image in ROI
            d_Ir_xy_roi = d_Ir_xy[i-w:i+w+1, j-w:j+w+1] # delta_Ir(x,y)
            
            dx = 1
            dy = 1
            
            x0 = [dx, dy]
            bnds = ((-3, 3), (-3, 3))
            #a = our_cost([dx, dy], I_xy_roi = I_xy_roi2, d_Ir_xy_roi = d_Ir_xy_roi2, Io = Io)
            #print(a)
            

            res = minimize(our_cost, x0, method='TNC', bounds = bnds, options={'gtol': 1e-20})
            
            #print(res)
            if mode_TD == 1:
                T, D = caculate_TD(d_Ir_xy_roi, I_xy_roi, window, dy, dx)
            
            
            mylist_T.append(T) 
            mylist_D.append(D)  
            mylist_dx.append(res['x'][0])
            mylist_dy.append(res['x'][1])
            
    return ({'T':mylist_T,'D':mylist_D,'dx':mylist_dx,'dy':mylist_dy})

def pad_zero_reshape_image_phase(image_list, reference_image, w):
    image_phase = np.reshape(image_list,[reference_image.shape[0]-2*w,reference_image.shape[1]-2*w])
    '''
    new_row = np.zeros((w,image_phase.shape[1]))
    image_phase = np.concatenate((image_phase,new_row))
    image_phase = np.concatenate((new_row,image_phase))

    new_col = np.zeros((image_phase.shape[0],w))
    image_phase = np.concatenate((image_phase,new_col),axis = 1)
    image_phase = np.concatenate((new_col,image_phase),axis = 1)
    '''
    
    return (image_phase)

def phase_retrieve (dx, dy, pix_w, d, lambda_s):
    nx = int(np.shape(dx)[0])
    ny = int(np.shape(dy)[1])

    theta_x = np.arctan(dx*pix_w/d)
    theta_y = np.arctan(dy*pix_w/d)

    if nx%2 == 0:
        # Kx = (-nx/2:nx/2-1)/(nx*pix_w) in Matlab
        Kx = [float(x)/(nx*pix_w) for x in range(-int(nx/2), int(nx/2))]       
        #Kx = Kx/(nx*pix_w)
        Kx[int(nx/2)+1] = -Kx[int(nx/2)+2]/1000
    else:
        Kx = [float(x)/2/(nx*pix_w) for x in range(-int(nx), int(nx), 2)]

    if ny%2 == 0:
        # Kx = (-nx/2:nx/2-1)/(nx*pix_w) in Matlab
        Ky = [float(x)/(ny*pix_w) for x in range(-int(ny/2), int(ny/2))]
        #Ky = Ky/(ny*pix_w)
        Ky[int(ny/2)+1] = -Ky[int(ny/2)+2]/1000
    else:
        Ky = [float(x)/2/(ny*pix_w) for x in range(-int(ny), int(ny), 2)]
    Kx = np.asarray(Kx)
    Ky = np.asarray(Ky)
    print('Kx shape:', np.shape(Kx))
    print('Ky shape:', np.shape(Ky))

    I_comb = []
    for i in range (np.shape(theta_x)[0]):
        for j in range (np.shape(theta_x)[1]):
            I_comb[i,j] = complex(theta_x[i, j], theta_y[i, j])
    print(I_comb)

    A = -np.tile(Ky[::-1],(nx,1))
    B = np.tile(Kx[::-1],(1,ny))
                 
    filter_LP = []
    for i in range (np.shape(theta_x)[0]):
        for j in range (np.shape(theta_x)[1]):
            filter_LP[i,j] = complex(A[i, j], B[i, j])
    filter_LP = 1/filter_LP
    filter_LP = np.fft.fftshift(filter_LP);
    print(filter_LP)

    FtI = np.fft.fft2(I_comb);
    argiFt = filter_LP*FtI;

    Phi = (2*math.pipi/lambda_s)*np.fft.ifft2(argiFt)/1000;

    return (Phi)
    

    

    
        

    
if __name__ == "__main__":
    # read data


    
    path = .. # file directory
    filename = .. # file name
    data = hdf5storage.loadmat(path+'\\'+filename, squeeze_me=True)
    print('original data has shape as:', np.shape(data['Is']))
    SP = data['Is'][2:-2,2:-2]# sandpaper
    SS = data['I'][2:-2,2:-2] #sandpaper and sample
    Abs = data['Io'][2:-2,2:-2] #obsorption only
    print('cropped data has shape as:', np.shape(SP))

#%%    
    fig1 = plt.figure()
    plt.subplot(121)
    plt.imshow(SP)
    plt.title('sandpaper')
    plt.subplot(122)
    plt.imshow(SS)
    plt.title('sample + sandpaper')

#%%   
    
    # fourier transform to determine window size
    img_size = np.shape(SP)
    print(img_size)
    line = SP[int(img_size[0]/2), 2:-2]-np.mean(SP[int(img_size[0]/2), 2:-2])
    print(np.shape(line))
    Ft_sp = np.fft.fftshift(abs(np.fft.fft(line)))
    fig2 = plt.figure()
    plt.subplot(121)
    plt.plot(line)
    plt.title('sandpaper')
    plt.subplot(122)
    plt.plot(Ft_sp)
    plt.show()

    print('maximum frequency:', np.argmax(Ft_sp))
    
    
    pixel_n = 8 # average pattern size
    global window
    window = create_window(pixel_n)
    
    print('window has shape:', np.shape(window))

  


    mode_TD = 1 # caculate T and D

    img_shape = np.shape(SS)
    print(img_shape)
    w = int((np.shape(window)[0]-1)/2)
    
    results_w = caculate_displacement (SS, SP, w)
    image_phase_x = pad_zero_reshape_image_phase(results_w['dx'], SS, w = w)
    image_phase_y = pad_zero_reshape_image_phase(results_w['dy'], SS, w = w)
    image_abs = pad_zero_reshape_image_phase(results_w['T'], SS, w = w)
    image_d= pad_zero_reshape_image_phase(results_w['D'], SS, w = w)



    plt.imshow(image_phase_x)
    plt.show()
    plt.imshow(image_phase_y)
    plt.show()


    
    

    
    

   
    

    





    
