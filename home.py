from itertools import count
import cv2;
import numpy ;
from tkinter import *
import tkinter as tk  
import fileinput
import tkinter.filedialog
from tkinter import filedialog 
from fileinput import filename 
from PIL import ImageTk, Image
from numpy.core.shape_base import hstack
from scipy.ndimage import maximum_filter, minimum_filter
from matplotlib import image, pyplot as plt
import os
from scipy.sparse.construct import vstack
from sklearn.cluster import MeanShift, estimate_bandwidth
from skimage.color import rgb2gray
from skimage import data
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from tkinter import messagebox
from sklearn.svm import SVC
import warnings
#from keras.preprocessing import image
# Data file load 
# from imutils import paths

class project:

    def __init__(self):
        self.input = ""
        window =tk.Tk()
        window.title("Machine Learning")
        self.createWindow(window)
        window.geometry('1000x1000')
        window.configure(bg='#00838f')
        window.mainloop()      

    def Browes(self):
            self.input1 = filedialog.askopenfilename(initialdir="/", title="Select A Photo",
                                   filetype=(("jpeg files", "*.jpg"), ("all files", "*.*"),))                       
            
            self.img = Image.open(self.input1)
            self.imgv = ImageTk.PhotoImage(self.img)
            self.lbl.configure(image=self.imgv)
            self.lbl.image =self.imgv 



    def Exit(self):
        exit()

        
    def Ideal_lowpass(self):
        img = cv2.imread(self.input1,0)
        dft = numpy.fft.fft2(img, axes=(0, 1))
        dft_shift = numpy.fft.fftshift(dft)
        mag = numpy.abs(dft_shift)
        spec = numpy.log(mag) / 20
        x, y = numpy.shape(img)
        midpointx, midpointy = x // 2, y // 2
        maskideal = numpy.zeros((x, y), numpy.uint8)
        Do = 50
        for u in range(0, x):
            for v in range(0, y):
                if numpy.sqrt(((u - midpointx) ** 2) + ((v - midpointy) ** 2)) <= Do:
                    maskideal[u][v] = 255
        pil_img = (dft_shift * maskideal) / 255
        pil_img = numpy.fft.ifftshift(pil_img)
        pil_img = numpy.fft.ifft2(pil_img, axes=(0, 1))
        pil_img = numpy.abs(pil_img).clip(0, 255).astype(numpy.uint8)
        print(pil_img) 
        total = numpy.hstack([img,pil_img])
        self.imgv = ImageTk.PhotoImage(Image.fromarray(total))
        self.lbl.configure(image=self.imgv)
        self.lbl.image =self.imgv  
        cv2.imwrite("test.jpg",pil_img)
        self.input1 ="test.jpg"


    def ideal_highpass(self):
        img = cv2.imread(self.input1,0)
        dft = numpy.fft.fft2(img, axes=(0, 1))
        dft_shift = numpy.fft.fftshift(dft)
        # generate spectrum from magnitude image (for viewing only)
        mag = numpy.abs(dft_shift)
        spec = numpy.log(mag) / 20
        x, y = numpy.shape(img)
        midpointx, midpointy = x // 2, y // 2
        maskideal = numpy.zeros((x, y), numpy.uint8)
        Do = 10
        for u in range(0, x):
            for v in range(0, y):
                if numpy.sqrt(((u - midpointx) ** 2) + ((v - midpointy) ** 2)) >= Do:
                    maskideal[u][v] = 255
        pil_img = (dft_shift * maskideal) / 255
        pil_img = numpy.fft.ifftshift(pil_img)
        pil_img = numpy.fft.ifft2(pil_img, axes=(0, 1))
        pil_img = numpy.abs(pil_img).clip(0, 255).astype(numpy.uint8)
        total = numpy.hstack([img,pil_img])
        self.imgv = ImageTk.PhotoImage(Image.fromarray(total))
        self.lbl.configure(image=self.imgv)
        self.lbl.image =self.imgv  
        cv2.imwrite("test.jpg",pil_img)
        self.input1 ="test.jpg"


    def im2double(self, im):
        max_val = numpy.max(im)
        min_val = numpy.min(im)
        return numpy.round((im.astype('float') - min_val) / (max_val - min_val) * 255)
    

    def Butterworth_highpass_filter(self):
        img = cv2.imread(self.input1, 0)
        dft = numpy.fft.fft2(img, axes=(0, 1))
        dft_shift = numpy.fft.fftshift(dft)
        # generate spectrum from magnitude image (for viewing only)
        mag = numpy.abs(dft_shift)
        spec = numpy.log(mag) / 20
        x, y = numpy.shape(img)
        midpointx, midpointy = x // 2, y // 2
        maskbutter = numpy.zeros((x, y))
        Do = 50
        for u in range(0, x):
            for v in range(0, y):
                duv = numpy.sqrt(((u - midpointx) ** 2) + ((v - midpointy) ** 2))
                maskbutter[u][v] = 1 / (1 + (Do / duv) ** 4)

        cv2.imshow("mask", maskbutter)
        maskbutter = self.im2double(maskbutter)
        pil_img = (dft_shift * maskbutter) / 255
        pil_img = numpy.fft.ifftshift(pil_img)
        pil_img = numpy.fft.ifft2(pil_img, axes=(0, 1))
        pil_img = numpy.abs(pil_img).clip(0, 255).astype(numpy.uint8)
        print(pil_img)
        total = numpy.hstack([img, pil_img])
        self.imgv = ImageTk.PhotoImage(Image.fromarray(total ))
        self.lbl.configure(image=self.imgv)
        self.lbl.image = self.imgv
        cv2.imwrite("test.jpg", pil_img)
        self.input1 = "test.jpg"



    def Butterworth_lowpass(self):
        img = cv2.imread(self.input1, 0)
        dft = numpy.fft.fft2(img, axes=(0, 1))
        dft_shift = numpy.fft.fftshift(dft)
        # generate spectrum from magnitude image (for viewing only)
        mag = numpy.abs(dft_shift)
        spec = numpy.log(mag) / 20
        x, y = numpy.shape(img)
        midpointx, midpointy = x // 2, y // 2
        maskbutter = numpy.zeros((x, y))
        Do = 30
        for u in range(0, x):
            for v in range(0, y):
                duv = numpy.sqrt(((u - midpointx) ** 2) + ((v - midpointy) ** 2))
                maskbutter[u][v] = 1 / (1 + (duv / Do) ** 4)
        maskbutter = self.im2double(maskbutter)
        pil_img = (dft_shift * maskbutter) / 255
        pil_img = numpy.fft.ifftshift(pil_img)
        pil_img = numpy.fft.ifft2(pil_img, axes=(0, 1))
        pil_img = numpy.abs(pil_img).clip(0, 255).astype(numpy.uint8)
        print(pil_img)
        total = numpy.hstack([img, pil_img])
        self.imgv = ImageTk.PhotoImage(Image.fromarray(total))
        self.lbl.configure(image=self.imgv)
        self.lbl.image = self.imgv
        cv2.imwrite("test.jpg", pil_img)
        self.input1 = "test.jpg"

    def GaussianHPF(self):
        img = cv2.imread(self.input1, 0)
        dft = numpy.fft.fft2(img, axes=(0, 1))
        dft_shift = numpy.fft.fftshift(dft)
        # generate spectrum from magnitude image (for viewing only)
        mag = numpy.abs(dft_shift)
        spec = numpy.log(mag) / 20
        x, y = numpy.shape(img)
        midpointx, midpointy = x // 2, y // 2
        maskgussine = numpy.zeros((x, y))
        Do = 30
        for u in range(0, x):
            for v in range(0, y):
                duv = numpy.sqrt(((u - midpointx) ** 2) + ((v - midpointy) ** 2))
                maskgussine[u][v] = 1 - numpy.exp((-(duv) ** 2) / (2 * (Do ** 2)))
        maskbutter = self.im2double(maskgussine)
        pil_img = (dft_shift * maskbutter) / 255
        pil_img = numpy.fft.ifftshift(pil_img)
        pil_img = numpy.fft.ifft2(pil_img, axes=(0, 1))
        pil_img = numpy.abs(pil_img).clip(0, 255).astype(numpy.uint8)
        print(pil_img)
        total = numpy.hstack([img, pil_img])
        self.imgv = ImageTk.PhotoImage(Image.fromarray(total ))
        self.lbl.configure(image=self.imgv)
        self.lbl.image = self.imgv
        cv2.imwrite("test.jpg", pil_img)
        self.input1 = "test.jpg"

    def GaussianLPF(self):
        img = cv2.imread(self.input1, 0)
        dft = numpy.fft.fft2(img, axes=(0, 1))
        dft_shift = numpy.fft.fftshift(dft)
        # generate spectrum from magnitude image (for viewing only)
        mag = numpy.abs(dft_shift)
        spec = numpy.log(mag) / 20
        x, y = numpy.shape(img)
        midpointx, midpointy = x // 2, y // 2
        maskgussine = numpy.zeros((x, y))
        Do = 30
        for u in range(0, x):
            for v in range(0, y):
                duv = numpy.sqrt(((u - midpointx) ** 2) + ((v - midpointy) ** 2))
                maskgussine[u][v] = numpy.exp((-(duv) ** 2) / (2 * (Do ** 2)))
        maskbutter = self.im2double(maskgussine)
        pil_img = (dft_shift * maskbutter) / 255
        pil_img = numpy.fft.ifftshift(pil_img)
        pil_img = numpy.fft.ifft2(pil_img, axes=(0, 1))
        pil_img = numpy.abs(pil_img).clip(0, 255).astype(numpy.uint8)
        print(pil_img)
        total = numpy.hstack([img, pil_img])
        self.imgv = ImageTk.PhotoImage(Image.fromarray(total ))
        self.lbl.configure(image=self.imgv)
        self.lbl.image = self.imgv
        cv2.imwrite("test.jpg", pil_img)
        self.input1 = "test.jpg"

    # def Butterworth_lowpass(self):
    #     img = cv2.imread(self.input1, 0)
    #     dft = numpy.fft.fft2(img, axes=(0, 1))
    #     dft_shift = numpy.fft.fftshift(dft)
    #     # generate spectrum from magnitude image (for viewing only)
    #     mag = numpy.abs(dft_shift)
    #     spec = numpy.log(mag) / 20
    #     x, y = numpy.shape(img)
    #     midpointx, midpointy = x // 2, y // 2
    #     maskbutter = numpy.zeros((x, y))
    #     Do = 30
    #     for u in range(0, x):
    #         for v in range(0, y):
    #             duv = numpy.sqrt(((u - midpointx) * 2) + ((v - midpointy) * 2))
    #             maskbutter[u][v] = 1 / (1 + (duv / Do) ** 4)
    #     maskbutter = self.im2double(maskbutter)
    #     pil_img = (dft_shift * maskbutter) / 255
    #     pil_img = numpy.fft.ifftshift(pil_img)
    #     pil_img = numpy.fft.ifft2(pil_img, axes=(0, 1))
    #     pil_img = numpy.abs(pil_img).clip(0, 255).astype(numpy.uint8)
    #     cv2.imshow("ee",pil_img) 
    #     cv2.imwrite("test.jpg", pil_img)
    #     self.input1 = "test.jpg"

    def search_for_file_path(self):
        tempdir = filedialog.askdirectory(initialdir="/", title='Please select a directory')
        if len(tempdir) > 0:
            print("You chose: %s" % tempdir)
        return tempdir
    def load_images_from_folder(self):
        folder = self.search_for_file_path()
        images = []
        for f in os.listdir(folder):   
            img = cv2.imread(os.path.join(folder, f))
            if img is not None:
                images.append(img)   
        self.dataset =images               
        return images





    def Arithmatic(self):
        img = cv2.imread(self.input1,0)
        #cv2.imshow("Orginal",img)
        w, h = img.shape
        img = img/numpy.max(img)
        amf = numpy.zeros_like(img)
        for row in range(w-1):
            for col in range(h-1):
                amf[row][col] = (1/9)*(img[row-1][col+1]+img[row][col+1]+img[row+1][col+1]+img[row-1]
                               [col]+img[row][col]+img[row+1][col]+img[row-1][col-1]+img[row][col-1]+img[row+1][col-1])
        amf = amf/numpy.max(amf)
        total = numpy.hstack([img,amf])
        self.imgv = ImageTk.PhotoImage(Image.fromarray(total*255))
        self.lbl.configure(image=self.imgv)
        self.lbl.image =self.imgv  
        cv2.imwrite("test.jpg",amf*255)
        self.input1 ="test.jpg"
       # cv2.imshow("Arithmatic",cv2.imread("test.jpg"))






    def Geometric_mean_filter(self):
        img = cv2.imread(self.input1,0)
        w,h = img.shape
        img = img/numpy.max(img)
        gmf= numpy.zeros_like(img)
        for r in range(w-1):
            for c in range(h-1):
                gmf[r][c] =((img[r-1][c+1]*img[r][c+1]*img[r+1][c+1]*img[r-1][c]*img[r][c]*img[r+1][c]*img[r-1][c-1]*img[r][c-1]*img[r+1][c-1]))**(1/9)
        total = numpy.hstack([img,gmf])
        self.imgv = ImageTk.PhotoImage(Image.fromarray(total*255))
        self.lbl.configure(image=self.imgv)
        self.lbl.image =self.imgv               
        cv2.imwrite("test.jpg",gmf*255)
        self.input1 ="test.jpg"
      
    def Harmonic_mean_filter(self):
        img = cv2.imread(self.input1,0)
        w,h = img.shape
        img =img/numpy.max(img)
        hmf =numpy.zeros_like(img)
        for r in range(w-1):
            for c in range(h-1):
                hmf[r][c]=9/(pow(img[r-1][c+1],-1)+ pow(img[r][c+1],-1)+pow(img[r+1][c+1],-1)+pow(img[r-1][c],-1)+pow(img[r][c],-1)+pow(img[r+1][c],-1)+pow(img[r-1][c-1],-1)+pow(img[r][c-1],-1)+pow(img[r+1][c-1],-1))
        total = numpy.hstack([img,hmf])
        self.imgv = ImageTk.PhotoImage(Image.fromarray(total*255))
        self.lbl.configure(image=self.imgv)
        self.lbl.image =self.imgv  
        cv2.imwrite("test.jpg",hmf*255)
        self.input1 ="test.jpg"
        

    def Contraharmonic_mean_filter(self):
        img = cv2.imread(self.input1,0)
        w,h = img.shape
        img = img/numpy.max(img)
        Cmf =numpy.zeros_like(img)
        Q = .001
        for row in range(w-1):
            for col in range(h-1):
                  Cmf[row][col]=(img[row-1][col+1]**(Q+1)+img[row][col+1]**(Q+1)+img[row+1][col+1]**(Q+1)+img[row-1][col]**(Q+1)+img[row][col]**(Q+1)+img[row+1][col]**(Q+1)+img[row-1][col-1]**(Q+1)+img[row][col-1]**(Q+1)+img[row+1][col-1]**(Q+1))/(img[row-1][col+1]**(Q)+img[row][col+1]**(Q)+img[row+1][col+1]**(Q)+img[row-1][col]**(Q)+img[row][col]**
                  (Q)+img[row+1][col]**(Q)+img[row-1][col-1]**(Q)+img[row][col-1]**(Q)+img[row+1][col-1]**(Q)) 

        total = numpy.hstack([img,Cmf])
        self.imgv = ImageTk.PhotoImage(Image.fromarray(total*255))
        self.lbl.configure(image=self.imgv)
        self.lbl.image =self.imgv  
        cv2.imwrite("test.jpg",img*255)
        self.input1 ="test.jpg"
        
    def max_filter(self):
        img = cv2.imread(self.input1,0)
        w,h = img.shape
        img = img/numpy.max(img)
        Maxfilter =numpy.zeros_like(img)
        for row in range(w-1):
            for col in range(h-1):
                arr = [img[row-1][col+1],img[row][col+1],img[row+1][col+1],img[row-1][col],img[row][col],img[row+1][col],img[row-1][col-1],img[row][col-1],img[row+1][col-1]]
                Maxfilter[row][col] =numpy.max(arr);
        total = numpy.hstack([img,Maxfilter])
        self.imgv = ImageTk.PhotoImage(Image.fromarray(total*255))
        self.lbl.configure(image=self.imgv)
        self.lbl.image =self.imgv  
        cv2.imwrite("test.jpg",Maxfilter*255)
        self.input1 ="test.jpg"
             

    def min_filter(self):
        img = cv2.imread(self.input1,0)
        w,h = img.shape
        img = img/numpy.max(img)
        Minfilter =numpy.zeros_like(img)
        for row in range(w-1):
            for col in range(h-1):
                arr = [img[row-1][col+1],img[row][col+1],img[row+1][col+1],img[row-1][col],img[row][col],img[row+1][col],img[row-1][col-1],img[row][col-1],img[row+1][col-1]]
                Minfilter[row][col] =numpy.min(arr);
        total = numpy.hstack([img,Minfilter])
        self.imgv = ImageTk.PhotoImage(Image.fromarray(total*255))
        self.lbl.configure(image=self.imgv)
        self.lbl.image =self.imgv  
        cv2.imwrite("test.jpg",Minfilter*255)
        self.input1 ="test.jpg"
              

    
    def median_filter(self):
        img = cv2.imread(self.input1,0)
        w,h = img.shape
        img = img/numpy.max(img)
        medianfilter =numpy.zeros_like(img)
        for row in range(w-1):
            for col in range(h-1):
                arr = [img[row-1][col+1],img[row][col+1],img[row+1][col+1],img[row-1][col],img[row][col],img[row+1][col],img[row-1][col-1],img[row][col-1],img[row+1][col-1]]
                medianfilter[row][col] =numpy.median(arr);
        total = numpy.hstack([img,medianfilter])
        self.imgv = ImageTk.PhotoImage(Image.fromarray(total*255))
        self.lbl.configure(image=self.imgv)
        self.lbl.image =self.imgv  
        cv2.imwrite("test.jpg",medianfilter*255)
        self.input1 ="test.jpg"
        
        
    # def midpoint(self):
    #     img = cv2.imread(self.input1, 0)
    #     w, c = img.shape
    #     img = img/numpy.max(img)
    #     new_image = numpy.zeros_like(img)
    #     maxf = maximum_filter(new_image, (3, 3))
    #     minf = minimum_filter(new_image, (3, 3))
    #     midpoint = (maxf + minf) / 2

    #     total = numpy.hstack([img, midpoint])
    #     cv2.imshow("Midpoint", total)
    #     cv2.imwrite("test.jpg", midpoint*255)
    #     self.input1 = "test.jpg"    

    def midpoint(self):
        img = cv2.imread(self.input1,0)
        w, c = img.shape
        img = img/numpy.max(img)
        new_image = numpy.zeros_like(img)
        maxf = maximum_filter(img, (3, 3))
        minf = minimum_filter(img, (3, 3))
        midpoint = (maxf + minf) / 2
        #cv2.imshow("MidPoint", midpoint)    
        total = numpy.hstack([img, midpoint])
        self.imgv = ImageTk.PhotoImage(Image.fromarray(total*255))
        self.lbl.configure(image=self.imgv)
        self.lbl.image =self.imgv  
        cv2.imwrite("test.jpg", midpoint*255)
        self.input1 = "test.jpg"    
    

    def Alpha_Trimgmed_Mean_Filter(self):
        img = cv2.imread(self.input1, 0)
        w, h = img.shape
        img = img/numpy.max(img)
        alpha = numpy.zeros_like(img)
        d = -1
        for row in range(w-1):
           for col in range(h-1):
              alpha[row][col] = (1/(3-d))*(img[row-1][col+1]+img[row][col+1]+img[row+1][col+1]+img[row-1]
                                     [col]+img[row][col]+img[row+1][col]+img[row-1][col-1]+img[row][col-1]+img[row+1][col-1])
        total = numpy.hstack([img, alpha])                             
        self.imgv = ImageTk.PhotoImage(Image.fromarray(total*255))
        self.lbl.configure(image=self.imgv)
        self.lbl.image =self.imgv  
        cv2.imwrite("test.jpg", alpha*255)
        self.input1 = "test.jpg"              
        


    # def Alpha_Trimmed(self,mask):
    #     img = cv2.imread(self.input1, 0)
    #     x, y = numpy.shape(img)  # Get num of Rows and Columns of Image
    #     m, n = numpy.shape(mask)
    #     # New Image With Bigger Size From Image
    #     new_image = numpy.ones((x + m - 1, y + n - 1), numpy.uint8)
    #     # Final Image To Reset New Values In It
    #     fin_image = numpy.ones((x, y), numpy.uint8)
    #     m = m // 2
    #     n = n // 2
    #     new_image[m:new_image.shape[0] - m, n:new_image.shape[1] -
    #             n] = self.input1  # Put Image In New Image
    #     for i in range(m, x + m):
    #         for j in range(n, y + n):
    #             avg = (new_image[i - m:i + m + 1, j - n:j + n +
    #                1].sum()) / (mask.shape[0] * mask.shape[1])
    #         fin_image[i - m, j - n] = int(avg)
    #     fin_image
    #     cv2.imshow(" ", Alpha_Trimmed(img, np.ones((5, 5), np.float32)))    




    def Global_Thresshold(self):
        image = cv2.imread(self.input1, 0)
        image = cv2.medianBlur(image,3)
        ret, thresh1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        ret, thresh2 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
        ret, thresh3 = cv2.threshold(image, 127, 255, cv2.THRESH_TRUNC)
        ret, thresh4 = cv2.threshold(image, 127, 255, cv2.THRESH_TOZERO)
        ret, thresh5 = cv2.threshold(image, 127, 255, cv2.THRESH_TOZERO_INV)
        title = ["Original Image", "Binary",
         "Binary_INV", "TRUNC", "TOZERO", "TOZERO_INV"]
        images = [image, thresh1, thresh2, thresh3, thresh4, thresh5]

        for i in range(6):
            plt.subplot(2, 3, i+1)
            plt.imshow(images[i], 'gray')
            plt.title(title[i])
            plt.xticks([]), plt.yticks([])

        plt.show()
    def Local_Adaptive_Thresholding(self):
        image =cv2.imread(self.input1,0)
        image = cv2.medianBlur(image, 3) 
        ret, thresh1 = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)
        thresh2 = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        thresh3 = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11,2)
        title = ["Original Image", "Global Thresholding",
                 "Adaptive Mean Thresholding", "Adaptive Gaussian Thresholding"]
        images = [image, thresh1, thresh2, thresh3]
        for i in range(4):
            plt.subplot(2, 2, i+1)
            plt.imshow(images[i], 'gray')
            plt.title(title[i])
            plt.xticks([]), plt.yticks([])
        plt.show()    

    def OTSU_Thresholding(self):
        img = cv2.imread(self.input1,0)
        ret, thresh2 = cv2.threshold(
       img, 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        new_result = cv2.medianBlur(thresh2, 3)
        total = numpy.hstack([img, new_result])                             
        self.imgv = ImageTk.PhotoImage(Image.fromarray(total*255))
        self.lbl.configure(image=self.imgv)
        self.lbl.image =self.imgv  
        cv2.imwrite("test.jpg", new_result*255)
        self.input1 = "test.jpg"  

    def Sobal(self):
        img = cv2.imread(self.input1,0)
        img_blur = cv2.GaussianBlur(img, (3, 3), 0)

        # Sobel Edge Detection
        sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0,
                          ksize=3)  # Sobel Edge Detection on the X axis
        sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1,
                          ksize=3)  # Sobel Edge Detection on the Y axiss
        # Combined X and Y Sobel Edge Detection
        sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=3)
        # Display Sobel Edge Detection Images
        total = numpy.hstack[img,sobelx,sobely,sobelxy]
        self.imgv = ImageTk.PhotoImage(Image.fromarray(total*255))
        self.lbl.configure(image=self.imgv)
        self.lbl.image =self.imgv  
    
    def Watershad(self):
        img = cv2.imread(self.input1) 
        img=cv2.resize(img,(512,512)) 
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
        ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU) 

        # cv2.imshow('Thesh',thresh) 
        # cv2.waitKey(0) 
        kernel = numpy.ones((3,3),numpy.uint8) 
        opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2) 

        # cv2.imshow('Morph',opening) 
        # cv2.waitKey(0) 
        sure_bg = cv2.dilate(opening,kernel,iterations=3) 
        # cv2.imshow('Sure bg',sure_bg) 
        # cv2.waitKey(0) 
        dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5) 
        ret, sure_fg = cv2.threshold(dist_transform,0.12*dist_transform.max(),255,0) 
        sure_fg = numpy.uint8(sure_fg) 
        # cv2.imshow('Sure fg',sure_fg) 
        # cv2.waitKey(0) 
        unknown = cv2.subtract(sure_bg,sure_fg) 
        # cv2.imshow('Subtract',unknown) 
        # cv2.waitKey(0) 
        ret, markers = cv2.connectedComponents(sure_fg) 
        markers = markers+1 
        markers[unknown==255] = 0 
        markers = cv2.watershed(img,markers) 
        img[markers == -1] = [0,255,0] 
        self.imgv = ImageTk.PhotoImage(Image.fromarray(img))
        self.lbl.configure(image=self.imgv)
        self.lbl.image =self.imgv  
    

    def Kmeans(self):
        img = cv2.imread(self.input1) 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        pixel_values = img.reshape((-1, 3)) 
        pixel_values = numpy.float32(pixel_values) 
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.95) 
        k = 3 
        _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS) 
        centers = numpy.uint8(centers) 
        labels = labels.flatten() 
        segmented_image = centers[labels.flatten()] 
        segmented_image = segmented_image.reshape(img.shape) 
        total  =hstack([img,segmented_image])
        self.imgv = ImageTk.PhotoImage(Image.fromarray(total))
        self.lbl.configure(image=self.imgv)
        self.lbl.image =self.imgv  

    def LevelSET(self):
        image = cv2.imread(self.input1,0)
        # Subtract image from mean of image
        image1 = image - numpy.mean(image)
        # Apply smoothing filter on image to reduce noise
        imSmooth = cv2.GaussianBlur(image1, (5, 5), 0)
        # Vziualize the results
        total = numpy.vstack([image,image1*255,imSmooth*255])
        self.imgv = ImageTk.PhotoImage(Image.fromarray(total))
        self.lbl.configure(image=self.imgv)
        self.lbl.image =self.imgv  






    def meanshift(self):
        originImg = cv2.imread(self.input1)
        originImg=cv2.resize(originImg,(250,250))
# Shape of original image    
        originShape = originImg.shape


# Converting image into array of dimension [nb of pixels in originImage, 3]
# based on r g b intensities    
        flatImg=numpy.reshape(originImg, [-1, 3])


# Estimate bandwidth for meanshift algorithm    
        bandwidth = estimate_bandwidth(flatImg, quantile=0.1, n_samples=100) 

#MeanShoft take bandwidth and number of seeding samples which is selected randomly from pixels   
        ms = MeanShift(bandwidth = bandwidth, bin_seeding=True)

# Performing meanshift on flatImg    
        ms.fit(flatImg)

# (r,g,b) vectors corresponding to the different clusters after meanshift    
        labels=ms.labels_
        print(labels)

# Remaining colors after meanshift    
        cluster_centers = ms.cluster_centers_    
        with numpy.printoptions(threshold=numpy.inf):
            print(cluster_centers)
# Finding and diplaying the number of clusters    
        labels_unique = numpy.unique(labels)    
        n_clusters_ = len(labels_unique)    
        print("number of estimated clusters : %d" % n_clusters_)    
# Displaying segmented image    
        segmentedImg = cluster_centers[numpy.reshape(labels, originShape[:2])]
        segmentedImg = numpy.uint8(segmentedImg )
        self.imgv = ImageTk.PhotoImage(Image.fromarray(segmentedImg*255))
        self.lbl.configure(image=self.imgv)
        self.lbl.image =self.imgv  
    
    def Snake(self):
        img = data.astronaut()
        img = rgb2gray(img)
        s = numpy.linspace(0, 2*numpy.pi, 400)
        r = 100 + 100*numpy.sin(s)
        c = 220 + 100*numpy.cos(s)
        init = numpy.array([r, c]).T
        snake = active_contour(gaussian(img, 3, preserve_range=False),
                               init, alpha=0.015, beta=10, gamma=0.001)
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(img, cmap=plt.cm.gray)
        ax.plot(init[:, 1], init[:, 0], '--r', lw=3)
        ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
        ax.set_xticks([]), ax.set_yticks([])
        ax.axis([0, img.shape[1], img.shape[0], 0])
        plt.show()
   
   
    def Connected_Component(self):
        img = cv2.imread(self.input1, 0)
        img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]
        num_lab, lab = cv2.connectedComponents(img)
        lab_hue = numpy.uint8(179 * lab / numpy.max(lab))
        blank_ch = numpy.ones_like(lab_hue) * 255
        lab_img = cv2.merge([lab_hue, blank_ch, blank_ch])
        lab_img = cv2.cvtColor(lab_img, cv2.COLOR_HSV2BGR)
        lab_img[lab_hue == 0] = 0
        self.imgv = ImageTk.PhotoImage(Image.fromarray(lab_img*255))
        self.lbl.configure(image=self.imgv)
        self.lbl.image =self.imgv  


    def chain(self):
        img = cv2.imread(self.input1, 0)
        rows, cols = img.shape
        result = numpy.zeros_like(img)
        for x in range(rows):
            for y in range(cols):
                if img[x, y] >= 70:
                    result[x, y] = 0
                else:
                    result[x, y] = 1
        plt.imshow(result)
        # plt.imshow(result, cmap='Greys')
        ## Discover the first point
        for i, row in enumerate(result):
            for j, value in enumerate(row):
                if value == 1:
                    start_point = (i, j)
                    break
                else:
                    continue
                    break

        directions = [0, 1, 2,
                      7, 3,
                      6, 5, 4]
        dir2idx = dict(zip(directions, range(len(directions))))
        # print(dir2idx)
        change_j = [-1, 0, 1,  # x or columns
                    -1, 1,
                    -1, 0, 1]

        change_i = [-1, -1, -1,  # y or rows
                    0, 0,
                    1, 1, 1]

        border = []
        chain = []

        curr_point = start_point
        for direction in directions:
            idx = dir2idx[direction]
            new_point = (start_point[0] + change_i[idx], start_point[1] + change_j[idx])
            if result[new_point] != 0:  # if is ROI
                border.append(new_point)
                chain.append(direction)
                curr_point = new_point
                break
        count = 0
        while curr_point != start_point:
            # figure direction to start search
            b_direction = (direction + 5) % 8
            dirs_1 = range(b_direction, 8)
            dirs_2 = range(0, b_direction)
            dirs = []
            dirs.extend(dirs_1)
            dirs.extend(dirs_2)
            for direction in dirs:
                idx = dir2idx[direction]
                new_point = (curr_point[0] + change_i[idx], curr_point[1] + change_j[idx])
                if result[new_point] != 0:  # if is ROI
                    border.append(new_point)
                    chain.append(direction)
                    curr_point = new_point
                    break
                if count == 1000:
                    break
            count += 1 
        self.imgv = ImageTk.PhotoImage(Image.fromarray(img))
        self.lbl.configure(image=self.imgv)
        self.lbl.image =self.imgv  
  
    # def identify_image(self,window):  # select image
    #   global canvas, IMG, image, image_tk
    #    canvas = tk.Canvas(window, width=500, height=450)
    #    canvas.place(x=300, y=100)
    # #   IMG = filedialog.askopenfilename()
    #   image = Image.open(IMG)
    #   image_tk = ImageTk.PhotoImage(image)
    #   image = image.resize((400, 380), Image.ANTIALIAS)
    #   image_tk = ImageTk.PhotoImage(image)
    #   canvas.create_image(0, 0, anchor='nw',image=image_tk)
          

#         def idea_high_pass(self):
#             img = cv2.imread(self.input1,0)  # load an image

# #     cv2.imshow("test", img)

#             plt.imshow(img, cmap='gray')
#             plt.show()
# #     Output is a 2D complex array. 1st channel real and 2nd imaginary
# # For fft in opencv input image needs to be converted to float32
#         dft = cv2.dft(numpy.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)

# # Rearranges a Fourier transform X by shifting the zero-frequency
# # component to the center of the array.
# # Otherwise it starts at the tope left corenr of the image (array)
#         dft_shift = numpy.fft.fftshift(dft)

# ##Magnitude of the function is 20.log(abs(f))
# # For values that are 0 we may end up with indeterminate values for log.
# # So we can add 1 to the array to avoid seeing a warning.
#         magnitude_spectrum = 20 * numpy.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))


# # lpf Circular LPF mask, center circle is 1, remaining all zeros
#         rows, cols = img.shape
#         crow, ccol = int(rows / 2), int(cols / 2)
#         mask = numpy.zeros((rows, cols, 2)), numpy.ui
#         cv2.imshow("ideal high pass", mask)


    def Butterworth_lowpass_Ds(self):
        for img in self.dataset:  
            img =cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)       
            dft = numpy.fft.fft2(img, axes=(0, 1))
            dft_shift = numpy.fft.fftshift(dft)
            # generate spectrum from magnitude image (for viewing only)
            mag = numpy.abs(dft_shift)
            spec = numpy.log(mag) / 20
            x, y = numpy.shape(img)
            midpointx, midpointy = x // 2, y // 2
            maskbutter = numpy.zeros((x, y))
            Do = 30
            for u in range(0, x):
                for v in range(0, y):
                    duv = numpy.sqrt(((u - midpointx) ** 2) + ((v - midpointy) ** 2))
                    maskbutter[u][v] = 1 / (1 + (duv / Do) ** 4)
            maskbutter = self.im2double(maskbutter)
            pil_img = (dft_shift * maskbutter) / 255
            pil_img = numpy.fft.ifftshift(pil_img)
            pil_img = numpy.fft.ifft2(pil_img, axes=(0, 1))
            pil_img = numpy.abs(pil_img).clip(0, 255).astype(numpy.uint8)
            print(pil_img)
            total = numpy.hstack([img, pil_img])
            cv2.imshow("BLS",total)
            cv2.waitKey()


    def Butterworth_highpass_Ds(self):
        for img in self.dataset: 
            img =cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
            dft = numpy.fft.fft2(img, axes=(0, 1))
            dft_shift = numpy.fft.fftshift(dft)
            # generate spectrum from magnitude image (for viewing only)
            mag = numpy.abs(dft_shift)
            spec = numpy.log(mag) / 20
            x, y = numpy.shape(img)
            midpointx, midpointy = x // 2, y // 2
            maskbutter = numpy.zeros((x, y))
            Do = 50
            for u in range(0, x):
                for v in range(0, y):
                    duv = numpy.sqrt(((u - midpointx) ** 2) + ((v - midpointy) ** 2))
                    maskbutter[u][v] = 1 / (1 + (Do / duv) ** 4)

            maskbutter = self.im2double(maskbutter)
            pil_img = (dft_shift * maskbutter) / 255
            pil_img = numpy.fft.ifftshift(pil_img)
            pil_img = numpy.fft.ifft2(pil_img, axes=(0, 1))
            pil_img = numpy.abs(pil_img).clip(0, 255).astype(numpy.uint8)
            print(pil_img)
            total = numpy.hstack([img, pil_img])
            cv2.imshow("BHS",total)
            cv2.waitKey()
    
    
    def GaussianHPF_Ds(self):
        for img in self.dataset: 
            img =cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
            dft = numpy.fft.fft2(img, axes=(0, 1))
            dft_shift = numpy.fft.fftshift(dft)
            # generate spectrum from magnitude image (for viewing only)
            mag = numpy.abs(dft_shift)
            spec = numpy.log(mag) / 20
            x, y = numpy.shape(img)
            midpointx, midpointy = x // 2, y // 2
            maskgussine = numpy.zeros((x, y))
            Do = 30
            for u in range(0, x):
                for v in range(0, y):
                    duv = numpy.sqrt(((u - midpointx) ** 2) + ((v - midpointy) ** 2))
                    maskgussine[u][v] = 1 - numpy.exp((-(duv) ** 2) / (2 * (Do ** 2)))
            maskbutter = self.im2double(maskgussine)
            pil_img = (dft_shift * maskbutter) / 255
            pil_img = numpy.fft.ifftshift(pil_img)
            pil_img = numpy.fft.ifft2(pil_img, axes=(0, 1))
            pil_img = numpy.abs(pil_img).clip(0, 255).astype(numpy.uint8)
            print(pil_img)
            total = numpy.hstack([img, pil_img])
            cv2.imshow("GHP",total)
            cv2.waitKey()

    def GaussianLPF_DS(self):
        for img in self.dataset: 
            img =cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
            dft = numpy.fft.fft2(img, axes=(0, 1))
            dft_shift = numpy.fft.fftshift(dft)
            # generate spectrum from magnitude image (for viewing only)
            mag = numpy.abs(dft_shift)
            spec = numpy.log(mag) / 20
            x, y = numpy.shape(img)
            midpointx, midpointy = x // 2, y // 2
            maskgussine = numpy.zeros((x, y))
            Do = 30
            for u in range(0, x):
                for v in range(0, y):
                    duv = numpy.sqrt(((u - midpointx) ** 2) + ((v - midpointy) ** 2))
                    maskgussine[u][v] = numpy.exp((-(duv) ** 2) / (2 * (Do ** 2)))
            maskbutter = self.im2double(maskgussine)
            pil_img = (dft_shift * maskbutter) / 255
            pil_img = numpy.fft.ifftshift(pil_img)
            pil_img = numpy.fft.ifft2(pil_img, axes=(0, 1))
            pil_img = numpy.abs(pil_img).clip(0, 255).astype(numpy.uint8)
            print(pil_img)
            total = numpy.hstack([img, pil_img])
            cv2.imshow("GLP",total)
            cv2.waitKey()

    def Arithmatic_dataset(self):
        for i in self.dataset:
            w, h,z = i.shape
            i = i/numpy.max(i)
            amf = numpy.zeros_like(i)
            for row in range(w-1):
                for col in range(h-1):
                    amf[row][col] = (1/9)*(i[row-1][col+1]+i[row][col+1]+i[row+1][col+1]+i[row-1]
                                   [col]+i[row][col]+i[row+1][col]+i[row-1][col-1]+i[row][col-1]+i[row+1][col-1])
            amf = amf/numpy.max(amf)
            total = numpy.hstack([i, amf])     
            cv2.imshow("Arithmatic",total)
            cv2.waitKey() 
        # for i in len(os.listdir(imglist)):
        #     plt.subplot(2, 3, i+1)
        #     plt.imshow(imglist[i], 'gray')
        #     plt.xticks([]), plt.yticks([])
        # plt.show()
    def Ideal_lowpass_DS(self):
        for i in self.dataset:
            i = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) 
            dft = numpy.fft.fft2(i, axes=(0, 1))
            dft_shift = numpy.fft.fftshift(dft)
            mag = numpy.abs(dft_shift)
            spec = numpy.log(mag) / 20
            x, y = i.shape
            midpointx, midpointy = x // 2, y // 2
            maskideal = numpy.zeros((x, y), numpy.uint8)
            Do = 50
            for u in range(0, x):
                for v in range(0, y):
                    if numpy.sqrt(((u - midpointx) ** 2) + ((v - midpointy) ** 2)) <= Do:
                        maskideal[u][v] = 255
                pil_img = (dft_shift * maskideal) / 255
                pil_img = numpy.fft.ifftshift(pil_img)
                pil_img = numpy.fft.ifft2(pil_img, axes=(0, 1))
                pil_img = numpy.abs(pil_img).clip(0, 255).astype(numpy.uint8)
            total = numpy.hstack([i, pil_img])
            cv2.imshow("Ideal lowpass filter", total)
            cv2.waitKey()
                # cv2.imwrite("test.jpg", pil_img)
                # self.input1 = "test.jpg"
    def ideal_highpass_DS(self):
        for i in self.dataset:
            i = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) 
            dft = numpy.fft.fft2(i, axes=(0, 1))
            dft_shift = numpy.fft.fftshift(dft)
            # generate spectrum from magnitude image (for viewing only)
            mag = numpy.abs(dft_shift)
            spec = numpy.log(mag) / 20
            x, y = numpy.shape(i)
            midpointx, midpointy = x // 2, y // 2
            maskideal = numpy.zeros((x, y), numpy.uint8)
            Do = 10
            for u in range(0, x):
                for v in range(0, y):
                   if numpy.sqrt(((u - midpointx) ** 2) + ((v - midpointy) ** 2)) >= Do:
                        maskideal[u][v] = 255
                pil_img = (dft_shift * maskideal) / 255
                pil_img = numpy.fft.ifftshift(pil_img)
                pil_img = numpy.fft.ifft2(pil_img, axes=(0, 1))
                pil_img = numpy.abs(pil_img).clip(0, 255).astype(numpy.uint8)
            total = numpy.hstack([i, pil_img])
            cv2.imshow("Ideal highpass filter",total )
            cv2.waitKey()

    def Geometric_mean_DS(self):
        for img in self.dataset:
            w, h,z = img.shape
            img = img / numpy.max(img)
            gmf = numpy.zeros_like(img)
            for r in range(w - 1):
                for c in range(h - 1):
                    gmf[r][c] = ((img[r - 1][c + 1] * img[r][c + 1] * img[r + 1][c + 1] * img[r - 1][c] * img[r][c] *
                              img[r + 1][c] * img[r - 1][c - 1] * img[r][c - 1] * img[r + 1][c - 1])) ** (1 / 9)
            total = numpy.hstack([img, gmf])
            cv2.imshow("GMF", total)
            cv2.waitKey()

    
    def Harmonic_mean_filter_DS(self):
        for img in self.dataset:
            w,h,z = img.shape
            img =img/numpy.max(img)
            hmf =numpy.zeros_like(img)
            for r in range(w-1):
                for c in range(h-1):
                    hmf[r][c]=9/(pow(img[r-1][c+1],-1)+ pow(img[r][c+1],-1)+pow(img[r+1][c+1],-1)+pow(img[r-1][c],-1)+pow(img[r][c],-1)+pow(img[r+1][c],-1)+pow(img[r-1][c-1],-1)+pow(img[r][c-1],-1)+pow(img[r+1][c-1],-1))
            total = numpy.hstack([img,hmf])
            cv2.imshow("HMF", total)
            cv2.waitKey()


    def Contraharmonic_mean_DS(self):
        for img in self.dataset:
            w, h,z = img.shape
            img = img / numpy.max(img)
            Cmf = numpy.zeros_like(img)
            Q = .001
            for row in range(w - 1):
                for col in range(h - 1):
                    Cmf[row][col] = (img[row - 1][col + 1] ** (Q + 1) + img[row][col + 1] ** (Q + 1) + img[row + 1][
                        col + 1] ** (Q + 1) + img[row - 1][col] ** (Q + 1) + img[row][col] ** (Q + 1) + img[row + 1][
                                         col] ** (Q + 1) + img[row - 1][col - 1] ** (Q + 1) + img[row][col - 1] ** (Q + 1) +
                                     img[row + 1][col - 1] ** (Q + 1)) / (
                                            img[row - 1][col + 1] ** (Q) + img[row][col + 1] ** (Q) + img[row + 1][
                                        col + 1] ** (Q) + img[row - 1][col] ** (Q) + img[row][col] **
                                            (Q) + img[row + 1][col] ** (Q) + img[row - 1][col - 1] ** (Q) + img[row][
                                                col - 1] ** (Q) + img[row + 1][col - 1] ** (Q))
            total = numpy.hstack([img, Cmf])
            cv2.imshow("Contraharmonic", total)
            cv2.waitKey()    

    

    def max_filter_DS(self):
        for img in self.dataset:
            w, h,z = img.shape
            img = img / numpy.max(img)
            Maxfilter = numpy.zeros_like(img)
            for row in range(w - 1):
                for col in range(h - 1):
                    arr = [img[row - 1][col + 1], img[row][col + 1], img[row + 1][col + 1], img[row - 1][col],
                           img[row][col], img[row + 1][col], img[row - 1][col - 1], img[row][col - 1],
                           img[row + 1][col - 1]]
                    Maxfilter[row][col] = numpy.max(arr);
            total = numpy.hstack([img, Maxfilter])
            cv2.imshow("Max", total)
            cv2.waitKey()

    def min_filter_DS(self):
        for img in self.dataset:
            w, h,z = img.shape
            img = img / numpy.max(img)
            Minfilter = numpy.zeros_like(img)
            for row in range(w - 1):
                for col in range(h - 1):
                    arr = [img[row - 1][col + 1], img[row][col + 1], img[row + 1][col + 1], img[row - 1][col],
                           img[row][col], img[row + 1][col], img[row - 1][col - 1], img[row][col - 1],
                           img[row + 1][col - 1]]
                    Minfilter[row][col] = numpy.min(arr);
            total = numpy.hstack([img, Minfilter])
            cv2.imshow("Min", total)
            cv2.waitKey()

    def median_filter_DS(self):
        for img in self.dataset:
            w, h,z = img.shape
            img = img / numpy.max(img)
            medianfilter = numpy.zeros_like(img)
            for row in range(w - 1):
                for col in range(h - 1):
                    arr = [img[row - 1][col + 1], img[row][col + 1], img[row + 1][col + 1], img[row - 1][col],
                           img[row][col], img[row + 1][col], img[row - 1][col - 1], img[row][col - 1],
                           img[row + 1][col - 1]]
                    medianfilter[row][col] = numpy.median(arr);
            total = numpy.hstack([img, medianfilter])
            cv2.imshow("Mediain", total)
            cv2.waitKey()

    def midpoint_DS(self):
        for img in self.dataset:
            w, c,z = img.shape
            img = img / numpy.max(img)
            new_image = numpy.zeros_like(img)
            maxf = maximum_filter(img, (3, 3))
            minf = minimum_filter(img, (3, 3))
            midpoint = (maxf + minf) / 2
            total = numpy.hstack([img, midpoint])
            cv2.imshow("Midpoint", total)
            cv2.waitKey()
        
    def Alpha_Trimgmed_Mean_Filter_DS(self):
        for img in self.dataset:
            w, h,z = img.shape
            img = img / numpy.max(img)
            alpha = numpy.zeros_like(img)
            d = -1
            for row in range(w - 1):
                for col in range(h - 1):
                    alpha[row][col] = (1 / (3 - d)) * (
                            img[row - 1][col + 1] + img[row][col + 1] + img[row + 1][col + 1] + img[row - 1]
                    [col] + img[row][col] + img[row + 1][col] + img[row - 1][col - 1] + img[row][col - 1] +
                            img[row + 1][col - 1])
            total = numpy.hstack([img, alpha])
            cv2.imshow("Alpha trimgmed", total)
            cv2.waitKey()

    def Global_Thresshold_DS(self):
        for image in self.dataset:
            image = cv2.medianBlur(image, 3)
            ret, thresh1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
            ret, thresh2 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
            ret, thresh3 = cv2.threshold(image, 127, 255, cv2.THRESH_TRUNC)
            ret, thresh4 = cv2.threshold(image, 127, 255, cv2.THRESH_TOZERO)
            ret, thresh5 = cv2.threshold(image, 127, 255, cv2.THRESH_TOZERO_INV)
            title = ["Original Image", "Binary",
                     "Binary_INV", "TRUNC", "TOZERO", "TOZERO_INV"]
            images = [image, thresh1, thresh2, thresh3, thresh4, thresh5]
            for i in range(6):
                plt.subplot(2, 3, i + 1)
                plt.imshow(images[i], 'gray')
                plt.title(title[i])
                plt.xticks([]), plt.yticks([])
            plt.show()

    
    def Local_Adaptive_Thresholding_DS(self):
        for image in self.dataset:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
            image = cv2.medianBlur(image, 3)
            ret, thresh1 = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)
            ret,thresh2 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
            ret,thresh3 = cv2.adaptiveThreshold(
                image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            title = ["Original Image", "Global Thresholding",
                     "Adaptive Mean Thresholding", "Adaptive Gaussian Thresholding"]
            images = [image, thresh1, thresh2, thresh3]
            for i in range(4):
                plt.subplot(2, 2, i + 1)
                plt.imshow(images[i])
                plt.title(title[i])
                plt.xticks([]), plt.yticks([])
            plt.show()


    def OTSU_Thresholding_DS(self):
        for img in self.dataset:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
            ret, thresh2 = cv2.threshold(
                img, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            new_result = cv2.medianBlur(thresh2, 3)
            total = numpy.hstack([img, new_result])
            cv2.imshow("Otsu Thresholding", total)
            cv2.waitKey()

    
    def Sobal_DS(self):
        for img in self.dataset:
            img_blur = cv2.GaussianBlur(img, (3, 3), 0)

            # Sobel Edge Detection
            sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0,
                               ksize=3)  # Sobel Edge Detection on the X axis
            sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1,
                               ksize=3)  # Sobel Edge Detection on the Y axiss
            # Combined X and Y Sobel Edge Detection
            sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=3)
            images= [img_blur,sobelx,sobely,sobelxy]
            title = ["Original image","Sobel X", "Sobel Y","SobeXY"]
            for i in range(4):
                plt.subplot(2, 2, i + 1)
                plt.imshow(images[i], 'gray')
                plt.title(title[i])
                plt.xticks([]), plt.yticks([])
            plt.show()


    def Watershad_DS(self):
        for img in self.dataset:
            img = cv2.resize(img,(450,450))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # cv2.imshow('Thesh',thresh)
            # cv2.waitKey(0)
            kernel = numpy.ones((3, 3), numpy.uint8)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

            # cv2.imshow('Morph',opening)
            # cv2.waitKey(0)
            sure_bg = cv2.dilate(opening, kernel, iterations=3)
            # cv2.imshow('Sure bg',sure_bg)
            # cv2.waitKey(0)
            dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
            ret, sure_fg = cv2.threshold(dist_transform, 0.12 * dist_transform.max(), 255, 0)
            sure_fg = numpy.uint8(sure_fg)
            # cv2.imshow('Sure fg',sure_fg)
            # cv2.waitKey(0)
            unknown = cv2.subtract(sure_bg, sure_fg)
            # cv2.imshow('Subtract',unknown)
            # cv2.waitKey(0)
            ret, markers = cv2.connectedComponents(sure_fg)
            markers = markers + 1
            markers[unknown == 255] = 0
            markers = cv2.watershed(img, markers)
            img[markers == -1] = [0, 255, 0]
            cv2.imshow('Result', img)
            cv2.waitKey()
    
    def Kmeans_DS(self):
        for img in self.dataset:
            img = cv2.resize(img,(450,450))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            pixel_values = img.reshape((-1, 3))

            pixel_values = numpy.float32(pixel_values)

            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.95)

            k = 3
            _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

            centers = numpy.uint8(centers)

            labels = labels.flatten()

            segmented_image = centers[labels.flatten()]

            segmented_image = segmented_image.reshape(img.shape)
            
            images= [img,segmented_image]
            title = ["Original image","Segmented image"]
            for i in range(2):
                plt.subplot(2, 2, i + 1)
                plt.imshow(images[i], 'gray')
                plt.title(title[i])
                plt.xticks([]), plt.yticks([])
            plt.show()

            

    def LevelSET_DS(self):
        for image in self.dataset:
            # Subtract image from mean of image
            image1 = image - numpy.mean(image)
            # Apply smoothing filter on image to reduce noise
            imSmooth = cv2.GaussianBlur(image1, (5, 5), 0)
            # Vziualize the results
            images= [image,image1,imSmooth]
            title = ["Original image","Image subtract mean", "Image after Subract"]
            for i in range(3):
                plt.subplot(2, 2, i + 1)
                plt.imshow(images[i], 'gray')
                plt.title(title[i])
                plt.xticks([]), plt.yticks([])
            plt.show()


    def meanshift_DS(self):
        for originImg in self.dataset:
            originImg = cv2.resize(originImg, (250, 250))
            # Shape of original image
            originShape = originImg.shape

            # Converting image into array of dimension [nb of pixels in originImage, 3]
            # based on r g b intensities
            flatImg = numpy.reshape(originImg, [-1,3])

            # Estimate bandwidth for meanshift algorithm
            bandwidth = estimate_bandwidth(flatImg, quantile=0.1, n_samples=100)

            # MeanShoft take bandwidth and number of seeding samples which is selected randomly from pixels
            ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)

            # Performing meanshift on flatImg
            ms.fit(flatImg)

            # (r,g,b) vectors corresponding to the different clusters after meanshift
            labels = ms.labels_
            #print(labels)

            # Remaining colors after meanshift
            cluster_centers = ms.cluster_centers_
            with numpy.printoptions(threshold=numpy.inf):
                print(cluster_centers)
            # Finding and diplaying the number of clusters
            labels_unique = numpy.unique(labels)
            n_clusters_ = len(labels_unique)
            print("number of estimated clusters : %d" % n_clusters_)
            # Displaying segmented image
            segmentedImg = cluster_centers[numpy.reshape(labels, originShape[:2])]
            segmentedImg = numpy.uint8(segmentedImg)
           
            total = hstack([originImg,segmentedImg])
            cv2.imshow('Mean shift', total)
            cv2.waitKey()


    def Snake_DS(self):
        for img in self.dataset:
            img = rgb2gray(img)
            s = numpy.linspace(0, 2 * numpy.pi, 400)
            r = 100 + 100 * numpy.sin(s)
            c = 220 + 100 * numpy.cos(s)
            init = numpy.array([r, c]).T
            snake = active_contour(gaussian(img, 3, preserve_range=False),
                                   init, alpha=0.015, beta=10, gamma=0.001)
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.imshow(img, cmap=plt.cm.gray)
            ax.plot(init[:, 1], init[:, 0], '--r', lw=3)
            ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
            ax.set_xticks([]), ax.set_yticks([])
            ax.axis([0, img.shape[1], img.shape[0], 0])
            plt.show()

    def Connected_Component_DS(self):
        for img in self.dataset:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
            img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]
            num_lab, lab = cv2.connectedComponents(img)
            lab_hue = numpy.uint8(179 * lab / numpy.max(lab))
            blank_ch = numpy.ones_like(lab_hue) * 255
            lab_img = cv2.merge([lab_hue, blank_ch, blank_ch])
            lab_img = cv2.cvtColor(lab_img, cv2.COLOR_HSV2BGR)
            lab_img[lab_hue == 0] = 0
            cv2.imshow("connected component",lab_img)
            cv2.waitKey()

    def chain_Ds(self):
        for img in self.dataset:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
            rows, cols = img.shape
            result = numpy.zeros_like(img)
            for x in range(rows):
                for y in range(cols):
                    if img[x, y] >= 70:
                        result[x, y] = 0
                    else:
                        result[x, y] = 1
            plt.imshow(result)
            # plt.imshow(result, cmap='Greys')
            ## Discover the first point
            for i, row in enumerate(result):
                for j, value in enumerate(row):
                    if value == 1:
                        start_point = (i, j)
                        break
                    else:
                        continue
                        break

            directions = [0, 1, 2,
                          7, 3,
                          6, 5, 4]
            dir2idx = dict(zip(directions, range(len(directions))))
            # print(dir2idx)
            change_j = [-1, 0, 1,  # x or columns
                        -1, 1,
                        -1, 0, 1]

            change_i = [-1, -1, -1,  # y or rows
                        0, 0,
                        1, 1, 1]

            border = []
            chain = []

            curr_point = start_point
            for direction in directions:
                idx = dir2idx[direction]
                new_point = (start_point[0] + change_i[idx], start_point[1] + change_j[idx])
                if result[new_point] != 0:  # if is ROI
                    border.append(new_point)
                    chain.append(direction)
                    curr_point = new_point
                    break
            count = 0
            while curr_point != start_point:
                # figure direction to start search
                b_direction = (direction + 5) % 8
                dirs_1 = range(b_direction, 8)
                dirs_2 = range(0, b_direction)
                dirs = []
                dirs.extend(dirs_1)
                dirs.extend(dirs_2)
                for direction in dirs:
                    idx = dir2idx[direction]
                    new_point = (curr_point[0] + change_i[idx], curr_point[1] + change_j[idx])
                    if result[new_point] != 0:  # if is ROI
                        border.append(new_point)
                        chain.append(direction)
                        curr_point = new_point
                        break
                    if count == 1000:
                        break
                count += 1
            plt.imshow(img, cmap='Greys')
            plt.plot([i[1] for i in border], [i[0] for i in border])
            plt.show()  

    def classification_R(self): 
        warnings.filterwarnings("ignore")
        cancer = datasets.load_breast_cancer()

        X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.33,random_state=109)
        clf = RandomForestClassifier(n_estimators = 10, random_state = 1)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        self.conf = metrics.confusion_matrix(y_test,y_pred)
        self.RAccuricy = metrics.accuracy_score(y_test, y_pred)*100
        tkinter.messagebox.showinfo("Accuricy",self.RAccuricy)
        print("##############Features####################3")
        print('feature',cancer.feature_names)
        print("##############Labels####################3")
        print('labels',cancer.target_names)
        print("##############Data####################3")
        print('data',cancer.data)
        print("##############Target Data####################3")
        print('target',cancer.target)
        print("##############Data Shape####################3")
        print('Shape',cancer.data.shape)
        print(self.conf);

    def classification_s(self): 
        warnings.filterwarnings("ignore")
        cancer = datasets.load_breast_cancer()
        X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.33,random_state=109)
        clf = SVC(kernel = "linear")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print("##############Features####################3")
        print('feature',cancer.feature_names)
        print("##############Labels####################3")
        print('labels',cancer.target_names)
        print("##############Data####################3")
        print('data',cancer.data)
        print("##############Target Data####################3")
        print('target',cancer.target)
        print("##############Data Shape####################3")
        print('Shape',cancer.data.shape)
       
        self.conf = metrics.confusion_matrix(y_test,y_pred)
        self.RAccuricy = metrics.accuracy_score(y_test, y_pred)*100
        tkinter.messagebox.showinfo("Accuricy",self.RAccuricy)
        print(self.conf);    
                 
    def createWindow(self,window):
        btn1 = tk.Button(window,text="Browse" ,width=40,command=self.Browes)
        btn1.grid(column=1,row=1)
        btn3 = tk.Button(window,text="Arithmatic" ,width=20,command=self.Arithmatic)
        btn3.grid(column=1,row=2)
        btn11 = tk.Button(window,text="Geomrtric" ,width=20,command=self.Geometric_mean_filter)
        btn11.grid(column=1,row=3)
        btn5 = tk.Button(window,text="Harmonic mean filter" ,width=20,command=self.Harmonic_mean_filter)
        btn5.grid(column=1,row=4)        
        btn6 = tk.Button(window,text="contraharmonic mean filter" ,width=20,command=self.Contraharmonic_mean_filter)
        btn6.grid(column=1,row=5) 
        btn7 = tk.Button(window,text="Max filter" ,width=20,command=self.max_filter)
        btn7.grid(column=1,row=6)
        btn8 = tk.Button(window,text="Min filter" ,width=20,command=self.min_filter)
        btn8.grid(column=1,row=7)
        btn9 = tk.Button(window,text="Median filter" ,width=20,command=self.median_filter)
        btn9.grid(column=1,row=8)
        btn10 = tk.Button(window,text="midpoint filter" ,width=20,command=self.midpoint)
        btn10.grid(column=1,row=9)
        btn11 = tk.Button(window,text="Alpha" ,width=20,command=self.Alpha_Trimgmed_Mean_Filter)
        btn11.grid(column=1,row=10)
        btn11 = tk.Button(window,text="Global Thresshold" ,width=20,command=self.Global_Thresshold)
        btn11.grid(column=1,row=11)
        btn12 = tk.Button(window,text="Local Thresshold" ,width=20,command=self.Local_Adaptive_Thresholding)
        btn12.grid(column=1,row=12)
        btn13 = tk.Button(window,text="OTUS Thresshold" ,width=20,command=self.OTSU_Thresholding)
        btn13.grid(column=1,row=13)
        btn14 = tk.Button(window,text="Sobel" ,width=20,command=self.Sobal)
        btn14.grid(column=1,row=14)  
        btn15 = tk.Button(window,text="Ideal low pass" ,width=20,command=self.Ideal_lowpass)
        btn15.grid(column=1,row=15)
        btn16 = tk.Button(window,text="Ideal high pass" ,width=20,command=self.ideal_highpass)
        btn16.grid(column=1,row=16)
        btn17 = tk.Button(window,text="Watershade" ,width=20,command=self.Watershad)
        btn17.grid(column=1,row=17)
        btn18 = tk.Button(window,text="Kmeans" ,width=20,command=self.Kmeans)
        btn18.grid(column=1,row=18)
        btn19 = tk.Button(window,text="Level Set" ,width=20,command=self.LevelSET)
        btn19.grid(column=1,row=19)
        btn20 = tk.Button(window,text="Mean shift" ,width=20,command=self.meanshift)
        btn20.grid(column=1,row=20)
        btn20 = tk.Button(window,text="Snake Algorithm" ,width=20,command=self.Snake)
        btn20.grid(column=1,row=21) 
        btn21 = tk.Button(window,text="Conected component" ,width=20,command=self.Connected_Component)
        btn21.grid(column=1,row=22)
        btn22 = tk.Button(window,text="Chain" ,width=20,command=self.chain)
        btn22.grid(column=1,row=23)

        btn23 = tk.Button(window, text="Butterworth high pass", width=20, command=self.Butterworth_highpass_filter)
        btn23.grid(column=1, row=24)
        
        btn24 = tk.Button(window, text="Butterworth low pass", width=20, command=self.Butterworth_lowpass)
        btn24.grid(column=1, row=25)

        btn25 = tk.Button(window, text="Gussian Highpass", width=20, command=self.GaussianHPF)
        btn25.grid(column=1, row=26)

        btn24 = tk.Button(window, text="Gussian lowpass", width=20, command=self.GaussianLPF)
        btn24.grid(column=1, row=27)
        

      
        # btn21 = tk.Button(window,text="Conected component" ,width=20,command=self.Connected_Component)
        # btn21.grid(column=1,row=23)
     
     
     
        btn100 = tk.Button(window,text="Browse Dataset" ,width=40,command=self.load_images_from_folder)
        btn100.grid(column=3,row=1)
        btn21 = tk.Button(window,text="ideal lowpass Ds" ,width=20,command=self.Ideal_lowpass_DS)
        btn21.grid(column=3,row=2) 
        btn21 = tk.Button(window,text="ideal highpass Ds" ,width=20,command=self.ideal_highpass_DS)
        btn21.grid(column=3,row=3)
        btn21 = tk.Button(window,text="Arithmatic mean Ds" ,width=20,command=self.Arithmatic_dataset)
        btn21.grid(column=3,row=4) 
        btn21 = tk.Button(window,text="Geometric mean Ds" ,width=20,command=self.Geometric_mean_DS)
        btn21.grid(column=3,row=5) 
        btn22 = tk.Button(window,text="Harmonic mean DS" ,width=20,command=self.Harmonic_mean_filter_DS)
        btn22.grid(column=3,row=6) 
        btn70 = tk.Button(window,text="Contraharmonic filter DS" ,width=20,command=self.Contraharmonic_mean_DS)
        btn70.grid(column=3,row=7)

        btn23 = tk.Button(window,text="Max filter DS" ,width=20,command=self.max_filter_DS)
        btn23.grid(column=3,row=8)
        btn24 = tk.Button(window,text="Min filter DS" ,width=20,command=self.min_filter_DS)
        btn24.grid(column=3,row=9)
        btn25 = tk.Button(window,text="Median filter DS" ,width=20,command=self.median_filter_DS)
        btn25.grid(column=3,row=10)
        btn26 = tk.Button(window,text="Midpoint filter DS" ,width=20,command=self.midpoint_DS)
        btn26.grid(column=3,row=11)        
        btn27 = tk.Button(window,text="Alpha Trimgmed DS" ,width=20,command=self.Alpha_Trimgmed_Mean_Filter_DS)
        btn27.grid(column=3,row=12)


        btn28 = tk.Button(window,text="Global Threshold" ,width=20,command=self.Global_Thresshold_DS)
        btn28.grid(column=3,row=13)

        btn29 = tk.Button(window,text="Adpaptive or local Threshold" ,width=20,command=self.Local_Adaptive_Thresholding_DS)
        btn29.grid(column=3,row=14)
        
        btn29 = tk.Button(window,text="Otsu Threshold" ,width=20,command=self.OTSU_Thresholding_DS)
        btn29.grid(column=3,row=15)

        btn30 = tk.Button(window,text="Level Set" ,width=20,command=self.LevelSET_DS)
        btn30.grid(column=3,row=16)
        
        btn30 = tk.Button(window,text="Snake Algorithm" ,width=20,command=self.Snake_DS)
        btn30.grid(column=3,row=17)
        btn30 = tk.Button(window,text="Watershad" ,width=20,command=self.Watershad_DS)
        btn30.grid(column=3,row=18)
        btn31 = tk.Button(window,text="Mean Shift" ,width=20,command=self.meanshift_DS)
        btn31.grid(column=3,row=19)
        btn31 = tk.Button(window,text="Kmean" ,width=20,command=self.Kmeans_DS)
        btn31.grid(column=3,row=20)
        btn31 = tk.Button(window,text="Sobel" ,width=20,command=self.Sobal_DS)
        btn31.grid(column=3,row=21)
        btn31 = tk.Button(window,text="Connected Component" ,width=20,command=self.Connected_Component_DS)
        btn31.grid(column=3,row=22)

        btn31 = tk.Button(window,text="Butterworth lowpass" ,width=20,command=self.Butterworth_lowpass_Ds)
        btn31.grid(column=3,row=23)

        
        btn32 = tk.Button(window,text="Butterworth Highpass" ,width=20,command=self.Butterworth_highpass_Ds)
        btn32.grid(column=3,row=24)

        btn33 = tk.Button(window,text="Gussian Highpass" ,width=20,command=self.GaussianHPF_Ds)
        btn33.grid(column=3,row=25)

        btn34 = tk.Button(window,text="Gussian Lowpass" ,width=20,command=self.GaussianLPF_DS)
        btn34.grid(column=3,row=26)
 
        btn34 = tk.Button(window,text="Chain" ,width=20,command=self.chain_Ds)
        btn34.grid(column=3,row=27)
        frame = tk.Frame(window,bg="#333", width=100,height=100)
        frame.place(x=900,y=100)
        self.lbl = tk.Label(frame)
        self.lbl.grid()

        btn40 = tk.Button(window,text="RandomForest Classification" ,width=40,command=self.classification_R)
        btn40.grid(column=2,row=5) 

        btn40 = tk.Button(window,text="SVM Classification" ,width=40,command=self.classification_s)
        btn40.grid(column=2,row=6) 
        btn40 = tk.Button(window,text="Exit" ,width=40,command=self.Exit)
        btn40.grid(column=2,row=7)
pro = project()