import torch
from torch import nn
import PIL
from PIL import Image as Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import vgg19, VGG19_Weights
import torch.nn.functional as function
import torch.optim as optim

        
def display_images(style_image_path:str, content_image:str):
    """Function to display the two images before processing

    Args:
        style_image_path (str): File path to the style image
        content_image (str): File path to the content image
    """
    image_one = Image.open(style_image_path)
    image_two = Image.open(content_image)

    plt.title("Style Image before Preprocessing")
    plt.imshow(image_one)
    plt.pause(3)

    plt.title("Content Image before Preprocessing")
    plt.imshow(image_two)
    plt.pause(3)


def pillow_transform(image_one_path:str, image_two_path:str):
    """Transforms an image path to a PIL image, then to a PyTorch tensor

    Args:
        image_one_path (str): _description_
        image_two_path (str): _description_

    Returns:
        _type_: _description_
    """
    image_one = Image.open(image_one_path)
    image_two = Image.open(image_two_path)

    tensor_transform = transforms.ToTensor()

    image_one = tensor_transform(image_one)
    image_two = tensor_transform(image_two)

    return image_one, image_two


def resize(style_image, content_image, image_size):
    """Resizes the two images to the same standard size

    Args:
        style_image (Pytorch Tensor): The style image
        content_image (Pytorch Tensor): The content image
    """

    transform = transforms.Resize(image_size)

    style_image = transform(style_image) 
    content_image = transform(content_image) 

    return style_image, content_image

def center_crop(style_image, content_image, crop_range=500):
    """Crops the center of the image

    Args:
        style_image (_type_): _description_
        content_image (_type_): _description_
        crop_range (int, optional): _description_. Defaults to 500.

    Returns:
        _type_: _description_
    """

    transform = transforms.CenterCrop(crop_range) 

    style_image = transform(style_image) 
    content_image = transform(content_image) 

    return style_image, content_image


def scale_pixel_values(style_image, content_image, mean, standard_deviation):
    """Scales the pixel values of the two images through a standardization
       proceduore of differencing the mean and dividing by the deviation

    Args:
        style_image (Pytorch Tensor): The style image
        content_image (Pytorch Tensor): The content image
    """
    mean = torch.tensor(mean).view(-1, 1, 1)
    standard_deviation = torch.tensor(standard_deviation).view(-1, 1, 1)
    transform = transforms.Normalize(mean=mean, std=standard_deviation, inplace=True)

    style_image = transform(style_image) 
    content_image = transform(content_image) 

    return style_image, content_image


def return_preprocessed_images(style_image, content_image, image_size, mean, standard_deviation, crop_range):
    """Applies all the preprocessing transformations and returns the adjusted images

    Args:
        style_image (_type_): _description_
        content_image (_type_): _description_
        image_size (_type_): _description_
        mean (_type_): _description_
        standard_deviation (_type_): _description_
        crop_range (_type_): _description_

    Returns:
        _type_: _description_
    """
    style_image, content_image = center_crop(style_image, content_image, crop_range)
    style_image, content_image = resize(style_image, content_image, image_size)
    style_image, content_image = scale_pixel_values(style_image, content_image, mean, standard_deviation)

    return style_image, content_image

class TransferNetwork(nn.Module):
    """Blank template to populate the model

    Args:
        nn (PyTorch module): Neural Network PyTorch Module
    """

    def __init__(self):
        super(TransferNetwork, self).__init__()


    def forward(self, input):
        #Store the feature maps at n-th layer
        outputs = {}
        
        #Pass input through each layer and store feature maps for convolutional layers
        for name, layer in self.named_children():
            input = layer(input)
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Sequential):  
                outputs[name] = input
        
        return outputs



def initialize_network(network=vgg19, newNetwork=TransferNetwork):
    """Initializing the neural network, pruning feedforward layer
       to only obtain the convolutional layers

    Args:
        network (_type_, optional): _description_. Defaults to resnet50.
        newNetwork (_type_, optional): _description_. Defaults to TransferNetwork.

    Returns:
        _type_: _description_
    """

    newNetwork = newNetwork()
    model = network(weights=VGG19_Weights.DEFAULT).features.eval()

    for idx, layer in enumerate(model.children()):
        print(layer)

        strLayer = str(layer).split('(')[0] + "_" + str(idx)
        #Breaking after the 8th Conv Layer
        if (strLayer == "ReLU_15"):
            break

        newNetwork.add_module(strLayer, layer)

    return newNetwork


def compute_style_loss(input_image, style_image, convolutional_network):

    input_image_batch = input_image.unsqueeze(0)
    style_image_batch = style_image.unsqueeze(0)
    
    loss = 0
    
    #Don't need the gradients on the feature maps, so we detach
    input_filters = convolutional_network(input_image_batch)
    style_filters = convolutional_network(style_image_batch)

    for layer in input_filters.keys():
        input_filters_loop = input_filters[layer].detach()
        style_filters_loop = style_filters[layer].detach()

        one, two, three, four = input_filters_loop.size()

        #Have to compute the gram matrix right here
        gram_input_filters = input_filters_loop.view(one * two, three * four)
        gram_input_filters = torch.mm(gram_input_filters, gram_input_filters.T)

        gram_style_filters = style_filters_loop.view(one * two, three * four)
        gram_style_filters = torch.mm(gram_style_filters, gram_style_filters.T)

        #Normalizing the gram matrix by dividing each element by the total number of elements in the matrix
        #This is necessary as large feature map sizes (three * four) will lead to large gram matrix values. These will weight
        #the early layers of the network more heavily (which we do not want)
        final_gram_input = gram_input_filters.div(one * two * three * four)
        final_gram_style = gram_style_filters.div(one * two * three * four)

        #Sum the losses at each layer
        loss += function.mse_loss(final_gram_input, final_gram_style)

    return loss


def compute_content_loss(input_image, content_image, convolutional_network):

    #Adding batch dimensions to images so we can pass through PyTorch Network
    content_image_batch = content_image.unsqueeze(0)
    input_image_batch = input_image.unsqueeze(0)
    
    #For content loss, we skip over the lower layers of the network
    #Each bottleneck block has 3 convolutional layers
        
    input_filters = convolutional_network(input_image_batch)
    content_filters = convolutional_network(content_image_batch)

    #For content loss, we only take higher level layers
    input_filters = input_filters["Conv2d_12"].detach()
    content_filters = content_filters["Conv2d_12"].detach()

    content_loss = function.mse_loss(input_filters, content_filters)

    return content_loss


def compute_total_loss(content_loss, style_loss, alpha, beta):
    """Computes the weighted sum loss

    Args:
        content_loss (_type_): _description_
        style_loss (_type_): _description_
        alpha (_type_): The weighting coefficient for the content loss
        beta (_type_): The weighting coefficient for the style loss

    """
    total_loss = (alpha * content_loss) + (beta * style_loss)

    return total_loss



def optimize_pixel_values(optimizer, optimization_steps, convolutional_network, input_image,
                          content_image, style_image, alpha, beta):
    """This function solves the optimization for minimizing the combined loss
    """
    #Requiring gradients for the input image so we can optimize the pixel values
    input_image.requires_grad_(True)

    #We are not training the parameters of the network
    convolutional_network.eval()
    convolutional_network.requires_grad_(False)

    steps = [0]
    
    while steps[0] < optimization_steps:


        def closure():
            #Ensure the pixel values of the input image remain between 0 and 1
            with torch.no_grad():
                input_image.clamp_(0, 1)

            #Zero out the gradients at each timestep to perform new optimization calculations after adjusting pixel values
            optimizer.zero_grad()

            content_loss = compute_content_loss(input_image, content_image, convolutional_network)
            style_loss = compute_style_loss(input_image, style_image, convolutional_network)
            total_loss = compute_total_loss(content_loss, style_loss, alpha, beta)
            print(f"Content loss is {content_loss} and style loss is {style_loss}")
            total_loss.backward()

            steps[0] += 1

            return total_loss


        optimizer.step(closure)

    #Clamping the final image's pixel values to between 0 and 1
    with torch.no_grad():
        input_image.clamp_(0,1)

    return input_image
        

def main_runner():
    """Runs the main algorithm to produce the output
    """
    pass

def plot_output(output_image, alpha, beta):
    plt.figure()
    plt.title(f"Ouput with Content Weight: {alpha} and Style Weight: {beta}")

    #Reformatting output image
    output_image = output_image.squeeze(0).detach().numpy()  
    output_image = output_image.transpose(1, 2, 0)  
    plt.imshow(output_image)
    plt.pause(100)




if __name__ == '__main__':
    initialize_network(vgg19, TransferNetwork)
    style_image = "Images\picasso.jpg"
    content_image = "Images\Pittsburgh.jpg"
    image_size = 128
    crop_range = 128
    optimization_steps = 10
    alpha = 1
    beta = 1000000
    #Using the canonical mean and standard deviation from ImageNet
    mean = [0.485, 0.456, 0.406]
    standard_deviation = [0.229, 0.224, 0.225]

    display_images(style_image, content_image)
    style_image, content_image = pillow_transform(style_image, content_image)
    style_image, content_image = return_preprocessed_images(style_image, content_image,
                                                            image_size, mean, standard_deviation, crop_range)
    
    input_image = content_image.clone()
    optimizer = optim.LBFGS([input_image])
    
    network = initialize_network()
    compute_content_loss(input_image, content_image, network)
    compute_style_loss(input_image, style_image, network)

    output_image = optimize_pixel_values(optimizer, optimization_steps, network, input_image,
                          content_image, style_image, alpha, beta)
    
    plot_output(output_image, alpha, beta)
    


