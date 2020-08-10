from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math

class bcb():
    def NN_interpolation(img,dstH,dstW):
        scrH,scrW,_=img.shape
        retimg=np.zeros((dstH,dstW,3),dtype=np.uint8)
        for i in range(dstH):
            for j in range(dstW):
                scrx=round((i+1)*(scrH/dstH))
                scry=round((j+1)*(scrW/dstW))
                retimg[i,j]=img[scrx-1,scry-1]
        return retimg

    def BiLinear_interpolation(img,dstH,dstW):
        scrH,scrW,_=img.shape
        img=np.pad(img,((0,1),(0,1),(0,0)),'constant')
        retimg=np.zeros((dstH,dstW,3),dtype=np.uint8)
        for i in range(dstH):
            for j in range(dstW):
                scrx=(i+1)*(scrH/dstH)-1
                scry=(j+1)*(scrW/dstW)-1
                x=math.floor(scrx)
                y=math.floor(scry)
                u=scrx-x
                v=scry-y
                retimg[i,j]=(1-u)*(1-v)*img[x,y]+u*(1-v)*img[x+1,y]+(1-u)*v*img[x,y+1]+u*v*img[x+1,y+1]
        return retimg

    def BiBubic(x):#产生16个像素点的权重
        x=abs(x)
        if x<=1:
            return 1-2*(x**2)+(x**3)
        elif x<2:
            return 4-8*x+5*(x**2)-(x**3)
        else:
            return 0

    def BiCubic_interpolation(self,img,dstH,dstW):#disH,W为目标函数的高和宽
        scrH,scrW,_=img.shape
        #img=np.pad(img,((1,3),(1,3),(0,0)),'constant')
        retimg=np.zeros((dstH,dstW,3),dtype=np.uint8)
        for i in range(dstH):
            for j in range(dstW):
                scrx=i*(scrH/dstH)
                scry=j*(scrW/dstW)
                x=math.floor(scrx)
                y=math.floor(scry)
                u=scrx-x
                v=scry-y
                tmp=0
                for ii in range(-1,2):
                    for jj in range(-1,2):
                        if x+ii<0 or y+jj<0 or x+ii>=scrH or y+jj>=scrW:
                            continue
                        tmp+=img[x+ii,y+jj]*self.BiBubic(ii-u)*self.BiBubic(jj-v)
                retimg[i,j]=np.clip(tmp,0,255)
        return retimg

    def function(self,k,data_path,save_path):
        image = np.array(Image.open(data_path))
        # image1 = NN_interpolation(image, image.shape[0] * k, image.shape[1] * k) #临近插值
        # image1 = Image.fromarray(image1.astype('uint8')).convert('RGB')
        # image1.save('./2.png')
        # image2 = BiLinear_interpolation(image, image.shape[0] * k, image.shape[1] * k) #双线性插值
        # image2 = Image.fromarray(image2.astype('uint8')).convert('RGB')
        # image2.save('./3.png')
        image3 = self.BiCubic_interpolation(self,image, image.shape[0] * k, image.shape[1] * k)  # 双三次插值
        image3 = Image.fromarray(image3.astype('uint8')).convert('RGB')
        image3.save(save_path)

