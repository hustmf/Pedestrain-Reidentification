from __future__ import print_function, absolute_import
from wxpy import *
import sys
import os
import numpy as np
import torch
from torch.autograd import Variable
from torch import nn
import pickle
from glob import glob
from PIL import Image  
import torchvision.transforms as transforms  
import models
from collections import OrderedDict

def load_model(fpath):
    if os.path.isfile(fpath):
        checkpoint = torch.load(fpath, map_location=lambda storage, loc: storage)
        print("=> Loaded checkpoint '{}'".format(fpath))
        return checkpoint
    else:
        raise ValueError("=> No checkpoint found at '{}'".format(fpath))
def extractFeature(model,imgpath):

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
def distance(queryfeature,referencefeatures,referenceimgs):

    x = torch.cat([queryfeature.unsqueeze(0)], 0)
    y = torch.cat([referencefeatures[f].unsqueeze(0) for f in referenceimgs], 0)
    m, n = x.size(0), y.size(0)
    y = y.view(n, -1)
    x = x.view(m, -1)

    dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist.addmm_(1, -2, x, y.t())
    return dist
def reid(imgpath,model):
        model.eval()
        referencefeatures = OrderedDict()
        working_dir= os.path.dirname(os.path.abspath(__file__))
        rePath=os.path.join(working_dir,'reference')
        referenceimgs=sorted(glob(os.path.join(rePath, '*.jpg')))
        referencefeatures=pickle.load(open(os.path.join(rePath,'rfeatures.txt'), 'rb'))
        queryfeatures = OrderedDict()
        queryfeatures[imgpath]=extractFeature(model,imgpath)
        L=[]
        for i in referenceimgs:
          filename=os.path.basename(i)
          L.append(filename)
        dist=distance(queryfeatures[imgpath],referencefeatures,L)
        distmat=dist.cpu().numpy()
        indices = np.argsort(distmat, axis=1)
        y=indices[0]
        imgPaths=[]
        for i in range(0,14):
            imgPaths.append(referenceimgs[y[i]])
        return imgPaths

def process():
    model = models.create('resnet50', num_features=128,dropout=0.5, num_classes=651)
    model = nn.DataParallel(model)#.cuda()
    working_dir=os.path.dirname(os.path.abspath(__file__))
    checkpoint = load_model(os.path.join(working_dir, 'model_best.pth.tar'))
    model.module.load_state_dict(checkpoint['state_dict'])
    #imgdir=working_dir
    imgpath='input.jpg'
    print(imgpath)
    imgPaths=reid(imgpath,model)
    print(imgPaths)
    return imgPaths

bot = Bot(console_qr=1)

my_friend = ensure_one(bot.friends().search(u'FZY'))
my_friend.send('Coming!')

@bot.register(my_friend)
def auto_reply(msg):
    msg.reply('Get')
    if msg.type != 'Picture' :
        msg.reply('Warning: Input should be a Pic')
    if msg.type == 'Picture' :
        Path = 'input.jpg'
        msg.get_file(save_path= Path)
        imgPaths=process()
        msg.reply('In Process')
        #msg.reply(msg.id)
       # msg.reply(imgPaths[0])
        for img in imgPaths:
            msg.reply_image(img)
        #return 'Get a pic'

embed()