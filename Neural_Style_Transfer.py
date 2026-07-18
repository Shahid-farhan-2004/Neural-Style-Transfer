import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms,models
from torchvision.models import VGG19_Weights
from PIL import Image
import matplotlib.pyplot as plt
import copy

transform=transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor()
])

def load_image(path):
    image=Image.open(path).convert('RGB')
    image=transform(image).unsqueeze(0)
    return image.to(torch.float)

#image that you want to change the style
content=load_image("")
#image of which style you want
style=load_image("")

vgg=models.vgg19(weights=VGG19_Weights.DEFAULT).features.eval()
for param in vgg.parameters():
    param.requires_grad=False
content_layer=['conv4_2']
style_layer=['conv1_1','conv2_1','conv3_1','conv4_1','conv5_1']

def get_features(x,model,layers):
    vgg_layer_names={
        '0':'conv1_1','5':'conv2_1','10':'conv3_1',
        '19':'conv4_1','21':'conv4_2','28':'conv5_1'
    }
    features={}
    for name,layer in model._modules.items():
        x=layer(x)
        if name in vgg_layer_names and vgg_layer_names[name] in layers:
            features[vgg_layer_names[name]]=x
    return features

def gram_matrix(tensor):
    b,c,h,w=tensor.size()
    features=tensor.view(c,h*w)
    g=torch.mm(features,features.t())
    return g/(c*h*w)

target=content.clone().requires_grad_(True)
optimizer=torch.optim.Adam([target],lr=0.001)

style_weight=1e5
content_weight=1

with torch.no_grad():
    style_features = get_features(style, vgg, style_layer)
    content_features = get_features(content, vgg, content_layer)
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_layer}
for step in range(300):
    target_features=get_features(target,vgg,style_layer+content_layer)
    content_loss=F.mse_loss(target_features['conv4_2'],content_features['conv4_2'])
    style_loss=0
    for layer in style_layer:
        target_gram=gram_matrix(target_features[layer])
        style_gram=style_grams[layer]
        style_loss+=F.mse_loss(target_gram,style_gram)
    total_loss=content_weight*content_loss+style_weight*style_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    if (step+1)%50==0:
        print(f"{step+1} the loss is {total_loss.item():.2f}")
output=target.detach().squeeze().permute(1,2,0).clamp(0,1).numpy()
plt.imshow(output)
plt.title("style image")
plt.show()


