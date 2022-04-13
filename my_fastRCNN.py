import os

import torch
import torchvision
from torchvision import transforms,datasets
from bs4 import BeautifulSoup
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import matplotlib.patches as patches

import abc

class generate_box:
    def __init__(self,xml_path,img_id):
        self.img_id = img_id
        self.xml_path = xml_path
        self.max_min_obj = ["xmin","ymin","xmax","ymax"]
        self.name = "name"
        self.lable = {"without_mask":0,"with_mask":1,"mask_weared_incorrect":2}
    def __get_label(self,obj):
        name = obj.find("name").text
        return self.lable[name]
    def __get_bounds(self,i):
        x = []
        for f in self.max_min_obj:
            x.append(int(i.find(f).text))
        return x
    def generate_target_label(self):
        with open(self.xml_path) as f:
            xml = f.read()
            self.__soup = BeautifulSoup(xml,"xml")
            self.objects = self.__soup.find_all("object")
            self.label_store = []
            self.annotiation_store = []
            for i in self.objects:
                self.label_store.append(self.__get_label(i))
                self.annotiation_store.append(self.__get_bounds(i))
        self.label_store = torch.as_tensor(self.label_store,dtype=torch.int64)# if we dont use int64, our model cant train in self.model([img],[annotation])
        self.annotiation_store = torch.as_tensor(self.annotiation_store,dtype=torch.float32)
        self.img_id = torch.as_tensor([self.img_id])# [self.img_id] NOT self.img_id
        return {"boxes":self.annotiation_store , "labels":self.label_store,"image_id": self.img_id}

class dataset_prepartion:
    def __init__(self,mask_img_path, mask_anno_path):
        self.transform  = self.__transformers()
        self.mask_img_path = mask_img_path
        self.mask_anno_path = mask_anno_path
        self.img = sorted(os.listdir(self.mask_img_path))
    def __transformers(self):
        return transforms.Compose([transforms.ToTensor()])
    def __getitem__(self, idx):
        file_image = 'maksssksksss' + str(idx) + '.png'
        file_label = 'maksssksksss' + str(idx) + '.xml'
        xml_path = os.path.join("mask/annotations",file_label)
        img_path = os.path.join("mask/images",file_image)
        img = self.transform(Image.open(img_path).convert("RGB"))
        T = generate_box(xml_path,idx)
        target = T.generate_target_label()
        return img , target
    def __len__(self):
        return len(self.img)
class model_value:
    def __init__(self, mask_img_path= "mask/images/",mask_anno_path = "mask/annotations/",epoch = 25,pretrain = True,num_class = 3):
        self.mask_img_path = mask_img_path
        self.mask_anno_path = mask_anno_path
        self.__pytorch_prepartion_values()
        self.epoch_number = epoch
        self.pretrain = pretrain
        self.num_classes =num_class
        self.model = self.__model_establishment()
        self.parameter = []
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(self.device)
    def __collate_fn(self,batch):
        return tuple(zip(*batch))
    def __pytorch_prepartion_values(self):
        self.dataset = dataset_prepartion(self.mask_img_path,self.mask_anno_path)
        self.data_loader = torch.utils.data.DataLoader(self.dataset, batch_size=4, collate_fn=self.__collate_fn)
        for imgs, annotations in self.data_loader:
            imgs = list(img.to(self.device) for img in imgs)
            self.annotations = [{k: v.to(self.device) for k, v in t.items()} for t in annotations]
            break
    def __model_establishment(self):
        detector =  torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=self.pretrain)
        infeature = detector.roi_heads.box_predictor.cls_score.in_features
        detector.roi_heads.box_predictor = FastRCNNPredictor(infeature, self.num_classes)
        return detector
    def train(self):
        for p in self.model.parameters():
            if p.requires_grad == True:
                self.parameter.append(p)
        self.optimizer = optim.SGD(self.parameter,lr=0.005,momentum=0.9,weight_decay=0.005)
        for epoch in range(self.epoch_number):
            self.model.train()
            i=0
            epoch_loss = 0
            for imgs , annotations in self.data_loader:
                i+=1
                imgs = list(img.to(self.device) for img in imgs)
                annotations = list([{k: v.to(self.device) for k, v in t.items()} for t in annotations])
                lossdict = self.model([imgs[0]],[annotations[0]])
                losses = sum(l for l in lossdict.values())
                self.optimizer.zero_grad()
                losses.backward()
                self.optimizer.step()
                epoch_loss+=losses
        print(epoch_loss)
    def test(self):
        self.model.eval()
        preds = self.model(self.imgs)
        print(preds)
        print("Prediction")
        self.plot_image(self.imgs[2], preds[2])
        print("Target")
        self.plot_image(self.imgs[2], self.annotations[2])
    def __plot_image(self,img_tensor, annotation):
        fig, ax = plt.subplots(1)
        img = img_tensor.cpu().data

        # Display the image
        ax.imshow(img.permute(1, 2, 0))

        for box in annotation["boxes"]:
            xmin, ymin, xmax, ymax = box

            # Create a Rectangle patch
            rect = patches.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), linewidth=1, edgecolor='r',
                                     facecolor='none')

            # Add the patch to the Axes
            ax.add_patch(rect)

        plt.show()

m= model_value()
m.test()


