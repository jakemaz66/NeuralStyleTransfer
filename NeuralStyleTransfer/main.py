from PIL import Image
from torchvision.models import vgg19, VGG19_Weights
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


#Use a GPU if available, and set the device type
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

#Set to eval because layers differ in behavior during evaluation and training
#VGG19 is split into two components: 'features' and 'classifier'. Features contains the 
#convolutional and pooling layers, which is what we want for neural-style transfer. Also,
#we are not altering the model's default weights so we set the mode to eval()
convolutional_network = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()

#Normalizing images based on the vgg19 network. The vgg19 network is trained on images
#with channels normalized by the following values
network_image_mean = torch.tensor([0.485, 0.456, 0.406])
network_image_std = torch.tensor([0.229, 0.224, 0.225])

#Setting the standard image size for content, style, and output images (larger sizes will take longer)
image_size_global = 512


def image_preprocess(image_name:str, image_size:int=image_size_global):
    """Preprocess the image by resizing it to a standard square shape and converting 
       the pixel values to a PyTorch tensor

    Args:
        image_name (str): The file path to the image
        image_size (int, optional): The number of pixels for the image

    Returns:
        tensor: A PyTorch tensor representation of the image
    """

    #Creating an object to resize the images to a common size and convert them to PyTorch tensors
    #transforms.Compose links together multiple transformations
    load = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])

    #Loading image with PIL and adding an extra dimension to the resulting tensor
    image = Image.open(image_name)
    image = load(image).unsqueeze(0)
    image.to(device, torch.float)

    return image


def display_images(style_image:torch.Tensor, content_image:torch.Tensor):
    """A function to display the content and style images

    Args:
        style_image (tensor): A tensor of the style image
        content_image (tensor): A tensor of the content image
    """

    unload = transforms.ToPILImage()

    #Transforming the images back to PIL for visualization
    image_one = style_image.cpu().clone()
    #Getting rid of the extra batch dimension
    image_one = image_one.squeeze(0)
    image_one = unload(image_one)
    
    #Transforming the images back to PIL for visualization
    image_two = content_image.cpu().clone()
    image_two = image_two.squeeze(0)
    image_two = unload(image_two)

    #Creating a subplot with one row and two columns
    figure, axes = plt.subplots(1, 2, figsize=(10,10))

    axes[0].imshow(image_one)
    axes[0].set_title("Style Image")
    
    axes[1].imshow(image_two)
    axes[1].set_title("Content Image")

    plt.show()


class content_image_loss(nn.Module):
    """Class to compute loss for the content of the image. It computes the distance between 
       the feature maps of the output image and content image

    Args:
        nn (PyTorch): nn module
    """
    
    def __init__(self, target,):

        super(content_image_loss, self).__init__()

        #Removing the target from the computational graph with detach()
        self.target = target.detach()


    def forward(self, input):
        """Computes a mean square error loss between the two feature maps

        Args:
            input (tensor): A tensor of the feature maps of the content image

        Returns:
            input: The same tensor
        """
        self.loss = F.mse_loss(input, self.target)
        
        return input
    

class style_image_loss(nn.Module):
    """Calculates loss for style, uses the gram matrix

    Args:
        nn (PyTorch): nn module
    """
    
    def __init__(self, target_feature):

        super(style_image_loss, self).__init__()

        #Target is the gram matrix of the features
        self.target = gram(target_feature).detach()


    def forward(self, input):
        """Computes the mean square error loss of the output image's gram matrix of kernels and
           the style images'

        Args:
            input (tensor): A tensor of feature maps, or kernels

        Returns:
            tensor: The same tensor
        """
        gram_matrix = gram(input)

        self.loss = F.mse_loss(gram_matrix, self.target)
        
        return input


def gram(input):
    """Computes the gram matrix of an input image

    Args:
        input (tensor): A PyTorch tensor representation of the image

    Returns:
        tensor: A returned gram matrix
    """
    #one is the batch dimension, which will be 1 (set with unsqueeze above), two is number of feature maps, 
    #three X four is each feature map size
    one, two, three, four = input.size()

    #Features will be a feature_maps by feature_map_size tensor
    features = input.view(one * two, three * four)

    #Obtaining the gram matrix by multiplying features by its transpose
    gram_product = torch.mm(features, features.T)

    #Normalizing the gram matrix by dividing each element by the total number of elements in the matrix
    #This is necessary as large feature map sizes (three * four) will lead to large gram matrix values. These will weight
    #the early layers of the network more heavily (which we do not want)
    final_gram = gram_product.div(one * two * three * four)

    return final_gram 


class normalize_images(nn.Module):
    """Creating a normalization module so we can insert it into our network at the beginning, 
       which will transform our images before they enter convolutional layers

    Args:
        nn (PyTorch): nn module
    """

    def __init__(self, mean, std):
        super(normalize_images, self).__init__()
        
        #Transforming the mean and standard deviation to a 3x1x1 tensor (for 3 channels in an image red, green, blue)
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)


    def forward(self, image):
        """Passing an image through the network outputs a normalized version of the image

        Args:
            image (tensor): A tensor representation of an image

        Returns:
            tensor: A normalized tensor
        """

        normalized_tensor = (image - self.mean) / self.std

        return normalized_tensor
    

def style_content_loss_layers(convolutional_network, network_mean, network_std, 
                              image_one, image_two):
    """Creates the model layers for computing the style and content losses.

    Args:
        convolutional_network (PyTorch network): Pre-trained convolutional network (e.g., VGG-19).
        network_mean (tensor): Normalization mean for the images.
        network_std (tensor): Normalization standard deviation for the images.
        image_one (tensor): The content image.
        image_two (tensor): The style image.

    Returns:
        nn.Sequential: A modified network model with content and style loss layers added.
        list: List of content losses.
        list: List of style losses.
    """

    #The content layers are higher level, abstract features, so we take the 4th convolutional layer of the VGG-19 network
    content_layers_default = ['convolutional_layer_4']

    #The style layers are given in multiple layers, as style features can be high or low level
    style_layers_default = ['convolutional_layer_1', 'convolutional_layer_2', 
                            'convolutional_layer_3', 'convolutional_layer_4', 
                            'convolutional_layer_5']

    #Creating normalization module and initializing the model with a normalization layer
    normalization = normalize_images(network_mean, network_std)
    model = nn.Sequential(normalization)

    content_loss_list = []
    style_loss_list = []

    counter = 0

    #Iterate through each layer of the convolutional network
    for layer in convolutional_network.children():

        #If it's a 2D convolutional layer
        if isinstance(layer, nn.Conv2d):
            counter += 1
            name = f'convolutional_layer_{counter}'

        #If it's an activation ReLU layer
        elif isinstance(layer, nn.ReLU):
            name = f'activation_layer_{counter}'

            # ReLU layers require `inplace=False` for loss layers to work correctly
            layer = nn.ReLU(inplace=False)

        #If it's a pooling layer
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pooling_layer_{counter}'

        #If it's a batch normalization layer
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'batchnorm_layer_{counter}'

        #Adding the current layer to the model with the appropriate name
        model.add_module(name, layer)

        #Add content loss after the specified content layers
        if name in content_layers_default:
            #The feature maps of the content image we want to match
            content_target_filters = model(image_one).detach()

            #Create the content loss layer
            content_loss = content_image_loss(content_target_filters)

            #Adding content loss layer to the model
            model.add_module(f'content_loss_{counter}', content_loss)

            #Append the content loss to the list
            content_loss_list.append(content_loss)

        #Add style loss after the specified style layers
        if name in style_layers_default:
            #The feature maps of the style image, using the Gram matrix
            style_target_filters = model(image_two).detach()

            #Create the style loss layer
            style_loss = style_image_loss(style_target_filters)

            #Adding style loss layer to the model
            model.add_module(f'style_loss_{counter}', style_loss)

            #Append the style loss to the list
            style_loss_list.append(style_loss)

    #Trim layers that come after the last loss layer (content or style)
    for excess_layer in range(len(model) - 1, -1, -1):
        if isinstance(model[excess_layer], content_image_loss) or isinstance(model[excess_layer], style_image_loss):
            break
        #Trim the model by removing excess layers
        model = model[:(excess_layer + 1)]

    return model, content_loss_list, style_loss_list


def get_input_image_optimizer(input_image:torch.tensor):
    """Limited Memory BFGS is an optimization algorithm. Computes the gradient of the loss function
       and asjusts the parameters (pixel values in this case) in the direction of steepest descent
       of the loss function.

    Args:
        input_image (torch.tensor): The input image as a tensor

    Returns:
        optimizer: The appropriate optimizer
    """
    optm = optim.LBFGS([input_image])

    return optm


def neural_style_transfer_algorithm(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=300,
                       style_weight=1000000, content_weight=1):
    """Runs the neural style transfer algorithm

    Args:
        cnn (PyTorch network): A convolutional neural network. We use the pre-trained vgg-19.
        normalizpation_mean (tensor): A tensor of values to normalize the image's channels
        normalization_std (tensor): A tensor of values to normalize the image's channels
        content_img (tensor): The content image (converted to a tensor)
        style_img (tensor): The style image (converted to a tensor)
        input_img (tensor): The input image (white noise which will get altered through gradient-descent losses)
        num_steps (int, optional): The number of steps to compute the losses. Defaults to 300.
        style_weight (int, optional): The weighting parameter to give to the style image. Defaults to 1000000.
        content_weight (int, optional): The weighting parameter to give to the content image. Defaults to 1.

    Returns:
        tensor: An ouput image that is a blend of the style and content images
    """


    print('Constructing the model...')
    model, content_losses, style_losses = style_content_loss_layers(cnn, normalization_mean, normalization_std,
                               content_img, style_img
    )

    input_img.requires_grad_(True)
    model.eval()
    model.requires_grad_(False)

    #Getting the optimizer to adjust pixel values for the input image
    optimizer = get_input_image_optimizer(input_img)

    print('Optimizing the input image to blend the content and style..')

    #Define run as an array due to Python scope. Arrays are mutable, whereas integers are not, within the 
    #inner function 'optimize'
    run = [0]

    while run[0] <= num_steps:

        def optimize():

            #Ensure the pixel values of the input image remain between 0 and 1
            with torch.no_grad():
                input_img.clamp_(0, 1)

            #Zero the gradients before each optimizer step to ensure non-stale gradients for each parameter
            #are used
            optimizer.zero_grad()

            #Passing input image through model
            model(input_img)

            #Retrieving content and style losses
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            #Total loss is a weighted combination of the content and style losses
            style_score *= style_weight
            content_score *= content_weight
            loss = style_score + content_score

            #Compute gradients
            loss.backward()

            run[0] += 1

            if run[0] % 20 == 0:

                print(f"Reached step {run}:")
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(optimize)

    #Ensuring final image has valid pixel values between 0 and 1
    with torch.no_grad():
        input_img.clamp_(0, 1)

    return input_img


if __name__ == '__main__':
    image_one_path = "Images\Surrealism.jpg"
    image_two_path = "Images\CollegeHall.jpg"
    white_noise = "Images\WhiteNoise.jpg"

    style_img = image_preprocess(image_one_path)
    content_img = image_preprocess(image_two_path)
    input_image = image_preprocess(white_noise)

    #Displaying images
    display_images(style_img, content_img)

    content_weight = 1
    style_weight = 1000000

    output_image = neural_style_transfer_algorithm(convolutional_network, network_image_mean, network_image_std,
                                                   content_img, style_img, input_image, num_steps=1000, content_weight=content_weight,
                                                   style_weight=style_weight)

    plt.figure()
    plt.title(f"Ouput with Content Weight: {content_weight} and Style Weight: {style_weight}")

    #Reformatting output image
    output_image = output_image.squeeze(0).detach().numpy()  
    output_image = output_image.transpose(1, 2, 0)  
    plt.imshow(output_image)
    plt.pause(100)








