import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
from torchvision.models import vgg19, VGG19_Weights

import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

#Set to eval because layers differ in behavior during evaluation and training
convolutional_network = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()

#Normalizing images based on the vgg19 network
network_mean = torch.tensor([0.485, 0.456, 0.406])
network_std = torch.tensor([0.229, 0.224, 0.225])


def image_preprocess(image_name:str, image_size:int=128):

    #Creating an object to resize the images to a common size and convert them to PyTorch tensors
    load = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])

    #Loading image with PIL and adding extra dimension to tensor
    image = Image.open(image_name)
    image = load(image).unsqueeze(0)
    image.to(device, torch.float)


def display_images(image_one:str, image_two:str):

    style_image = image_preprocess(image_one)
    content_image = image_preprocess(image_two)

    unload = transforms.ToPILImage()

    plt.ion()

    #Transforming the images back to PIL for visualization
    image_one = style_image.cpu().clone()
    image_one = image_one.squeeze(0)
    image_one = unload(image_one)
    plt.figure()
    plt.imshow(image_one)


class content_image_loss(nn.Module):
    
    def __init__(self, target,):

        super(content_image_loss, self).__init__()

        self.target = target.detach()
        self.loss = None


    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        
        return input


def gram(input):
    #one is the batch dimension, which will be 0, two is number of feature maps, three X four is each feature map size
    one, two, three, four = input.size()

    features = input.view(one * two, three * four)

    #Obtaining the gram matrix by multiplying features by its transpose
    gram_product = torch.mm(features, features.T)

    #Normalizing
    final_gram = gram_product.div(one * two * three * four)

    return final_gram 


class style_image_loss(nn.Module):
    
    def __init__(self, target_feature):

        super(style_image_loss, self).__init__()

        self.target = gram(target_feature).detach()
        self.loss = None


    def forward(self, input):
        gram_matrix = gram(input)

        self.loss = F.mse_loss(gram_matrix, self.target)
        
        return input
    

class normalize_images(nn.Module):

    def __init__(self, mean, std):
        super(normalize_images, self).__init__()

        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)


    def forward(self, image):
        normalized = (image - self.mean) / self.std

        return normalized
    

def style_content_loss_layers(convultional_network, network_mean, network_std, image_one, image_two):

    content_layers_default = ['conv_4']

    style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    normalization = normalize_images(network_mean, network_std)

    content_loss_list = []
    style_loss_list = []

    model = nn.Sequential(normalization)

    i = 0
    for layer in convultional_network.children():

        #If it's a 2d convolutional layer
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)

        #If it's an activation ReLU layer
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)

            layer = nn.ReLU(inplace=False)

        #If it's a pooling layer
        elif isinstance(layer, nn.MaxPool2d):
            name = 'maxPool_{}'.format(i)

        #If it's a batch normalization layer
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'batchNorm_{}'.format(i)

        #Adding the correct layer to the whole model
        model.add_module(name, layer)

        #Insert content loss layer after the content layer
        if name in content_layers_default:
            target = model(image_one).detach()

            content_loss = content_image_loss(target)

            model.add_module('content_loss_{}'.format(i), content_loss)

            #Append current content loss to list
            content_loss_list.append(content_loss)

        #Insert content loss layer after the content layer
        if name in style_layers_default:
            target = model(image_two).detach()

            style_loss = style_image_loss(target)

            model.add_module('style_loss_{}'.format(i), style_loss)

            #Append current content loss to list
            style_loss_list.append(style_loss)


    #Have to trim the extra layers
    for i in range(len(model) - 1, -1, -1):

        #If the current layer is a loss layer
        if isinstance(model[i], content_image_loss) or isinstance(model[i], style_image_loss):
            break 
        
        #Take everything up to and including the loss layer
        model = model[:(i + 1)]

    return model, content_loss_list, style_loss_list



def get_input_image_optimizer(input_image:torch.tensor):
    optm = optim.lbfgs([input_image])

    return optm



def neural_style_transfer_algorithm(convolutional_network, network_mean, network_std, image_one, 
                                    image_two, input_image, steps=300, style_weight=1000000, content_weight=1):
    
    model, content_losses, style_losses = style_content_loss_layers(convolutional_network, network_mean, network_std, image_one, image_two)


    input_image.requires_grad_(True)
    
    model.eval()
    model.requires_grad_(False)

    optimizer = get_input_image_optimizer(input_image)
    run = [0]

    while run[0] < steps:

        def closure():

            #Ensure that optimized values fall into normalized pixel range of 0 to 1
            with torch.no_grad():
                input_image.clamp(0, 1)

            optimizer.zero_grad()
            model(input_image)

            style = 0
            content = 0

            for loss in style_losses:
                style += loss 

            for loss in content_losses:
                content += loss 

            style += style_weight
            content += content_weight

            total_loss = style + content
            total_loss.backward()

            run[0] += 1

            return style + content 
        
        optimizer.step(closure)

    with torch.no_grad():
        input_image.clamp_(0, 1)

    return input_image


if __name__ == '__main__':
    image_one = "Images\ChuckNorris.jfif"
    image_two = "Images\StarryNight.jfif"
    input_image = image_one.clone()

    #Displaying images
    display_images(image_one, image_two)

    output_image = neural_style_transfer_algorithm(convolutional_network, network_mean, network_std,
                                                   image_one, image_two, input_image)

    plt.figure()
    plt.imshow(output_image, title='Output Image')

    plt.ioff()
    plt.show()









