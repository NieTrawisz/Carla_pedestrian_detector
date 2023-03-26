import matplotlib.pyplot as plt
import cv2
import scipy.ndimage.filters as filters
import numpy as np
import math
from skimage import transform
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
import os
import pickle

def imshow(img):
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def readim(prefix,number,ext):
    img=cv2.imread(prefix+'%06d.'%number+ext)
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def get_grad(im):
    dx = filters.convolve1d(np.int32(im), np.array([-1, 0, 1]), 1)
    dy = filters.convolve1d(np.int32(im), np.array([-1, 0, 1]), 0)
    return dx,dy

def hog(img,cellsize=8,num_parts=9):
    #%%getting mag and ang
    Rgrad=get_grad(img[:,:,0])
    Ggrad=get_grad(img[:,:,1])
    Bgrad=get_grad(img[:,:,2])
    
    mag=np.array([np.sqrt(dx*dx+dy*dy) for dx,dy in [Rgrad,Ggrad,Bgrad]])
    angle=np.array([np.arctan2(dy,dx) for dx,dy in [Rgrad,Ggrad,Bgrad]])*180/np.pi
    amax=np.argmax(mag,axis=0)
    
    magmax=np.zeros((mag.shape[1],mag.shape[2]),mag.dtype)
    angmax=np.zeros((angle.shape[1],angle.shape[2]),angle.dtype)
    for i in range(3):
        X,Y=np.where(amax==i)
        magmax[X,Y]=mag[i,X,Y]
        angmax[X,Y]=angle[i,X,Y]
    angmax[np.where(angmax<0)]+=180
    #%% cells
    YY,XX,_=img.shape
    YY_cell=YY//cellsize
    XX_cell=XX//cellsize
    
    HoG=np.zeros((YY_cell,XX_cell,num_parts))
    part_size=180/num_parts
    for y in range(0,YY_cell):
        for x in range(0,XX_cell):
            Y=y*cellsize
            X=x*cellsize
            local_mag=magmax[Y:Y+cellsize,X:X+cellsize]
            local_ang=angmax[Y:Y+cellsize,X:X+cellsize]
            
            #indexes of histogram
            ind=local_ang//part_size
            ind[np.where(ind==num_parts)]=num_parts-1
            ind=ind.astype('int8')
            
            #getting centers
            centers=ind*part_size+part_size/2
            
            #adding to histogram
            diff=local_ang-centers
            for yy in range(cellsize):
                for xx in range(cellsize):
                    if diff[yy,xx]>0:
                        if ind[yy,xx]+1<num_parts:
                            HoG[y,x,ind[yy,xx]+1]+=diff[yy,xx]/part_size*local_mag[yy,xx]
                        else:
                            HoG[y,x,0]+=diff[yy,xx]/part_size*local_mag[yy,xx]
                        HoG[y,x,ind[yy,xx]]+=(part_size-diff[yy,xx])/part_size*local_mag[yy,xx]
                    else:
                        HoG[y,x,ind[yy,xx]-1]+=-diff[yy,xx]/part_size*local_mag[yy,xx]
                        HoG[y,x,ind[yy,xx]]+=(part_size+diff[yy,xx])/part_size*local_mag[yy,xx]
    #%% Normalization in block
    e = math.pow(0.00001,2)
    F = []
    for jj in range(0,YY_cell-1):
        for ii in range(0,XX_cell-1):
            H0 = HoG[jj,ii,:]
            H1 = HoG[jj,ii+1,:]
            H2 = HoG[jj+1,ii,:]
            H3 = HoG[jj+1,ii+1,:]
            H = np.concatenate((H0, H1, H2, H3))
            n = np.linalg.norm(H)
            Hn = H/np.sqrt(math.pow(n,2)+e)
            F = np.concatenate((F,Hn))
    return F
    
#%%getting data

if __name__=='__main__':
    #training
    root_dir='images/'
    empty_len=len(os.listdir(root_dir+'empty_pedestrians/'))
    pedestrians_len=len(os.listdir(root_dir+'pedestrian/'))
    hog = cv2.HOGDescriptor()

    HOG_data = np.zeros([(empty_len+pedestrians_len),3781],np.float32)
    for i in range(0,empty_len):
        image = readim(root_dir+'empty_pedestrians/',i,'png')
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        hog_val=hog.compute(image)
        HOG_data[i,0] = 0
        HOG_data[i,1:] = hog_val

    for i in range(empty_len,pedestrians_len+empty_len):
        image = readim(root_dir+'pedestrian/',i-empty_len,'png')
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        hog_val=hog.compute(image)
        HOG_data[i,0] = 1
        HOG_data[i,1:] = hog_val

    labels = HOG_data[:,0]
    data = HOG_data[:,1:]
        
    data_train, data_test, labels_train,labels_test=train_test_split(data, labels, test_size=0.2, random_state=42)
    norm=Normalizer().fit(data_train)
    data_train=norm.transform(data_train)
    data_test=norm.transform(data_test)
    #%% svm validation
    clf = svm.SVC(kernel='linear', C =1.0, probability=True)
    clf.fit(data_train, labels_train)

    lp_train = clf.predict(data_train)
    print("Training accuracy: ",metrics.accuracy_score(labels_train,lp_train)*100,'%')
    print(metrics.confusion_matrix(labels_train,lp_train))

    lp_test = clf.predict(data_test)
    print("Test accuracy: ",metrics.accuracy_score(labels_test,lp_test)*100,'%')
    print(metrics.confusion_matrix(labels_test,lp_test))

    pickle.dump(clf, open('pedestrian_svm.bin', 'wb'))
    pickle.dump(norm, open('pedestrian_normalizer.bin', 'wb'))