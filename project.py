from __future__ import print_function, absolute_import
from PyQt5 import QtWidgets, QtGui
import sys
import os
import numpy as np
import torch
#import unicode
from torch.autograd import Variable
from torch import nn
import pickle
from glob import glob
from PIL import Image  
import torchvision.transforms as transforms  
from form import Ui_Form    # 导入生成form.py里生成的类
import models
from collections import OrderedDict
import xml.dom.minidom  
import xml.etree.ElementTree as ET

def load_model(fpath):#加载模型
    if os.path.isfile(fpath):
        checkpoint = torch.load(fpath, map_location=lambda storage, loc: storage)
        print("=> Loaded checkpoint '{}'".format(fpath))
        return checkpoint
    else:
        raise ValueError("=> No checkpoint found at '{}'".format(fpath))

def extractFeature(model,imgpath):#提取特征

    img=Image.open(imgpath).convert('RGB')  
    img=img.resize((128,256))  
    transformImg=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])
    IMG=transformImg(img)
    IMG.resize_(1,3,256,128)
   # model.eval()
    IMG= Variable(IMG, volatile=True)
    outputs = model(IMG)
    outputs = outputs.data.cpu()
    return outputs

def extractFeatureSet(model,imgdir):#提取目录下图片特征
    featureSet=OrderedDict()
    if os.path.isfile(os.path.join(imgdir,'rfeatures.txt')):
          featureSet=pickle.load(open(os.path.join(imgdir,'rfeatures.txt'), 'rb'))#第一次载入reference图片时，将该行注释，恢复上面五行，并保存
          print('1')
    else:
          i=0
          imgs=sorted(glob(os.path.join(imgdir, '*.jpg')))
          for imgpath in imgs:
              print(i/len(imgs)*100)#进度
              i=i+1
              imgname=os.path.basename(imgpath)
              featureSet[imgname]=extractFeature(model,imgpath)
          pickle.dump(featureSet, open(os.path.join(imgdir,'rfeatures.txt'), 'wb'))
    return featureSet

def distance(queryfeature,referencefeatures,referenceimgs):#计算欧氏距离

    x = torch.cat([queryfeature.unsqueeze(0)], 0)
    y = torch.cat([referencefeatures[f].unsqueeze(0) for f in referenceimgs], 0)
    m, n = x.size(0), y.size(0)
    y = y.view(n, -1)
    x = x.view(m, -1)

    dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist.addmm_(1, -2, x, y.t())
    return dist

def GenerateXml(filepath,imageName,result):#生成xml
    if os.path.isfile(filepath):
        tree=ET.parse(filepath)
        root = tree.getroot()
        e=ET.Element('Item',{'imageName':imageName})
        e.text=result
        for items in root.iter('Items'):
          items.append(e)
        tree.write(filepath)
    else:
        impl = xml.dom.minidom.getDOMImplementation()
        dom = impl.createDocument(None, 'Message', None)
        root = dom.documentElement  
        Items = dom.createElement('Items')
        root.appendChild(Items)
        
        Item=dom.createElement('Item')
        result=dom.createTextNode(result)
        Item.appendChild(result)
        Item.setAttribute("imageName",imageName) #增加属性
        Items.appendChild(Item)

        f= open(filepath, 'w')
        dom.writexml(f, addindent='  ', newl='\n')
        f.close()  
    return
class mywindow(QtWidgets.QWidget,Ui_Form):    
    
    working_dir= os.path.dirname(os.path.abspath(__file__))
    model = models.create('resnet50', num_features=128,dropout=0.5, num_classes=651)
    model = nn.DataParallel(model)#.cuda()
    checkpoint = load_model(os.path.join(working_dir, 'model_best.pth.tar'))
    model.module.load_state_dict(checkpoint['state_dict'])
    model.eval()
    def __init__(self):    
        super(mywindow,self).__init__()    
        self.setupUi(self)

    #定义槽函数
    def openQueryImg(self):#打开图片
        filename = QtWidgets.QFileDialog.getOpenFileName(self,"open file dialog","D:","Images (*.png *.xpm *.jpg)")
        print(filename)
        filename=filename[0]
        self.queryImgPath.setText(filename)
        png=QtGui.QPixmap(filename)
        png=png.scaledToHeight(self.queryImg.height())
        self.queryImg.setPixmap(png)
        self.filepath=filename
        return
    def reid(self):#检索图片
        imgpath=self.queryImgPath.toPlainText()
        if imgpath=="":
            return
        referencefeatures = OrderedDict()
        if self.referencePath.toPlainText()=="":
            return
        else:
            rePath=self.referencePath.toPlainText()
        print(rePath)
        if os.path.isdir(rePath):
            referencefeatures=extractFeatureSet(self.model,rePath)
        else:
            return
       
        referenceimgs=sorted(glob(os.path.join(rePath, '*.jpg')))
        queryfeatures = OrderedDict()
        queryfeatures[imgpath]=extractFeature(self.model,imgpath)
        referenceNames=[]
        for img in referenceimgs:
            imgname=os.path.basename(img)
            referenceNames.append(imgname)
        dist=distance(queryfeatures[imgpath],referencefeatures,referenceNames)
        distmat=dist.cpu().numpy()
        indices = np.argsort(distmat, axis=1)
        y=indices[0]
        png=QtGui.QPixmap(referenceimgs[y[0]])
        png=png.scaledToHeight(self.result1.height())
        self.result1.setPixmap(png)
        png=QtGui.QPixmap(referenceimgs[y[1]])
        png=png.scaledToHeight(self.result2.height())
        self.result2.setPixmap(png)
        png=QtGui.QPixmap(referenceimgs[y[2]])
        png=png.scaledToHeight(self.result3.height())
        self.result3.setPixmap(png)
        png=QtGui.QPixmap(referenceimgs[y[3]])
        png=png.scaledToHeight(self.result4.height())
        self.result4.setPixmap(png)
        png=QtGui.QPixmap(referenceimgs[y[4]])
        png=png.scaledToHeight(self.result5.height())
        self.result5.setPixmap(png)
        png=QtGui.QPixmap(referenceimgs[y[5]])
        png=png.scaledToHeight(self.result6.height())
        self.result6.setPixmap(png)
        png=QtGui.QPixmap(referenceimgs[y[6]])
        png=png.scaledToHeight(self.result7.height())
        self.result7.setPixmap(png)
        png=QtGui.QPixmap(referenceimgs[y[7]])
        png=png.scaledToHeight(self.result8.height())
        self.result8.setPixmap(png)
        png=QtGui.QPixmap(referenceimgs[y[8]])
        png=png.scaledToHeight(self.result9.height())
        self.result9.setPixmap(png)
        png=QtGui.QPixmap(referenceimgs[y[9]])
        png=png.scaledToHeight(self.result10.height())
        self.result10.setPixmap(png)
        png=QtGui.QPixmap(referenceimgs[y[10]])
        png=png.scaledToHeight(self.result11.height())
        self.result11.setPixmap(png)
        png=QtGui.QPixmap(referenceimgs[y[11]])
        png=png.scaledToHeight(self.result12.height())
        self.result12.setPixmap(png)
        png=QtGui.QPixmap(referenceimgs[y[12]])
        png=png.scaledToHeight(self.result13.height())
        self.result13.setPixmap(png)
        png=QtGui.QPixmap(referenceimgs[y[13]])
        png=png.scaledToHeight(self.result14.height())
        self.result14.setPixmap(png)
        png=QtGui.QPixmap(referenceimgs[y[14]])
        png=png.scaledToHeight(self.result15.height())
        self.result15.setPixmap(png)

        return

    def importReDir(self):
        dir_path=QtWidgets.QFileDialog.getExistingDirectory(self,"choose directory","D:")  
       # dir_path=unicode(dir_path.toUtf8(), 'utf-8', 'ignore')    
        self.referencePath.setText(dir_path)
        return
    def importQuDir(self):#导入query目录
        dir_path=QtWidgets.QFileDialog.getExistingDirectory(self,"choose directory","D:")  
     #   dir_path=unicode(dir_path.toUtf8(), 'utf-8', 'ignore')    
        self.queryPath.setText(dir_path)
        return
    def reidAll(self):
        if self.referencePath.toPlainText()=="" or self.queryPath.toPlainText()=="":
            return

        referencefeatures = OrderedDict()
        print(self.referencePath.toPlainText())
        if self.referencePath.toPlainText()=="":
            rePath=os.path.join(self.working_dir,'reference')
        else:
            rePath=self.referencePath.toPlainText()
        print(rePath)
        referenceimgs = sorted(glob(os.path.join(rePath, '*.jpg')))
        i=0
        referencefeatures=extractFeatureSet(self.model,rePath)

        L=[]#提取reference文件名用于xml写入
        referenceNames=[]
        for i in referenceimgs:
         # print(i)
          (filepath,tempfilename) = os.path.split(i)
          (filename,extension) = os.path.splitext(tempfilename)
          L.append(filename+' ')
          referenceNames.append(os.path.basename(i))

        queryfeatures = OrderedDict()
        if self.queryPath.toPlainText()=="":
            quPath=os.path.join(self.working_dir,'query')
        else:
            quPath=self.queryPath.toPlainText()
        queryimgs = sorted(glob(os.path.join(quPath, '*.jpg')))
        
        if os.path.isfile(os.path.join(self.working_dir,'result.xml')):
           os.remove(os.path.join(self.working_dir,'result.xml'))
        for imgpath in queryimgs:
           queryfeatures[imgpath]=extractFeature(self.model,imgpath)
           print(imgpath)
           dist=distance(queryfeatures[imgpath],referencefeatures,referenceNames)
           distmat=dist.cpu().numpy()
           indices = np.argsort(distmat, axis=1)
           y=indices[0]
           (filepath,tempfilename) = os.path.split(imgpath)
           (filename,extension) = os.path.splitext(tempfilename)
        
           #将该项写入xml中
           resultrank=[L[i]  for i in y] 
           resultrank=''.join(resultrank)
           GenerateXml(os.path.join(self.working_dir,'result.xml'),filename,resultrank)
        self.progressBar.setValue(100)
        return




app = QtWidgets.QApplication(sys.argv)
window = mywindow()
window.show()
sys.exit(app.exec_())

