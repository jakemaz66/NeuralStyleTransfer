from PIL import Image
from torchvision.models import vgg19, VGG19_Weights
import torch
import torch.nn as nn
import torch.nn.functional as function
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.transforms import functional as tranfunc


class content_image_loss(nn.Module):
    """Class to compute loss for the content of the image. It computes the distance between 
       the feature maps of the output image and content image. Inherits from the nn.Module
       class so we can insert it as a layer into the network.

    Args:
        nn (PyTorch): nn module from PyTorch
    """
    
    def __init__(self, content_feature_maps,):

        super(content_image_loss, self).__init__()

        #Removing the target from the computational graph with detach()
        self.content_feature_maps = content_feature_maps.detach()


    def forward(self, current_feature_maps):
        """Computes a mean square error loss between the two feature maps
           of the content image and the current iteration of the input image

        Args:
            input (tensor): A tensor of the feature maps of the current input image

        Returns:
            input: The same tensor
        """
        self.loss = function.mse_loss(current_feature_maps, self.content_feature_maps)
        
        return current_feature_maps
    

class style_image_loss(nn.Module):
    """Calculates loss for style, uses the difference of gram matrixces of 
       feature maps at a certain layer to compute loss. Inherits from the nn.Module
       class so we can insert it as a layer into the network.

    Args:
        nn (PyTorch): nn module from PyTorch
    """
    
    def __init__(self, style_feature_maps):

        super(style_image_loss, self).__init__()

        #Target is the gram matrix of the features
        self.style_feature_maps = compute_gram_matrix(style_feature_maps).detach()


    def forward(self, current_feature_maps):
        """Computes the mean square error loss of the output image's gram matrix of kernels and
           the style images'

        Args:
            input (tensor): A tensor of feature maps, or kernels

        Returns:
            tensor: The same tensor
        """

        #Computing gram matrix of current feature maps and getting mse loss
        gram_matrix = compute_gram_matrix(current_feature_maps)
        self.loss = function.mse_loss(gram_matrix, self.style_feature_maps)
        
        return current_feature_maps
    

class normalize_images(nn.Module):
    """Creating a normalization module so we can insert it into our network at the beginning, 
       which will transform our images before they enter convolutional layers. Thus, we can
       simply feed in images and they will automatically be processed appropriately.

    Args:
        nn (PyTorch): nn module from PyTorch
    """

    def __init__(self, mean, std):
        #Calling parent constructor from nn.Module
        super(normalize_images, self).__init__()
        
        #Transforming the mean and standard deviation to a 3x1x1 tensor (for 3 channels in an image red, green, blue)
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.standard_deviation = torch.tensor(std).view(-1, 1, 1)


    def forward(self, image):
        """Passing an image through the network outputs a normalized version of the image

        Args:
            image (tensor): A tensor representation of an image

        Returns:
            tensor: A normalized tensor
        """

        #Standardization x - mean / standard deviation
        normalized_tensor = (image - self.mean) / self.standard_deviation

        return normalized_tensor
    
class ManualLBFGS:
    """A Manual implementation of the LBFGS Algorithm. LBFGS approximates the second-derivative matrix
       for better convergence compared to vanilla gradient descent optimization. The 'curvature' 
       information gained allows the algorithm to take larger optimization steps in flatter 
       regions and smaller steps in steep regions. L stands for limited memory, as the algorithm
       approximates the derivates using the past n updates.
    """

    def __init__(self, params, learning_rate=0.0001, history_size=20):
        """Initializing optimizer class

        Args:
            params (vary): The thing to be optimized (input image)
            learning_rate (int, optional): The step size in gradient descent. Defaults to 0.0001.
            history_size (int, optional): The number of steps to go back in time. Defaults to 20.
        """
        #Storing the parameters
        self.params = params

        #How long to loop back through previous iterations
        self.history_size = history_size

        #The differences in parameters, gradients, and scaling weights
        self.parameter_differences = []
        self.gradient_differences = []
        self.scaling = []

        #The step size in adjusting the parameters (pixel values)
        self.learning_rate= learning_rate

        #For first step, we do not have amy parameters or gradients
        self.prev_params = None
        self.prev_grad = None


    def step(self, optimize):
        """This is the "step" in minimizing the loss, adjusting the pixel values of the input image.
           We adjust in the opposite direction of gradient of loss.

        Args:
            optimize (function): A function that returns the loss

        Returns:
            float: The loss
        """

        #Get the loss (which also calculates gradients because optimize() calls loss.backward())
        loss_at_step = optimize()
        
        #Retrieve the current gradients and parameters
        gradients = self._get_flat_grad()
        parameters = self._get_flat_params()

        #If we aren't the first step
        if self.prev_grad is not None:
            
            #Find the difference in the previous params and gradients. When differences are larger then the
            #slope of loss function is also greater
            diff_param = parameters - self.prev_params
            diff_grad = gradients - self.prev_grad

            #Scaling (add small number to prevent divide by 0)
            scaling_step = 1.0 / (diff_grad.dot(diff_param) + 0.00000001)

            #If we reached the max history size, pop an element to make room for previous iteration
            if len(self.parameter_differences) >= self.history_size:

                self.parameter_differences.pop(0)
                self.gradient_differences.pop(0)
                self.scaling.pop(0)

            #Otherwise add the current differences
            self.parameter_differences.append(diff_param)
            self.gradient_differences.append(diff_grad)
            self.scaling.append(scaling_step)

            #Retrieve the direction to adjust pixel values in based on LBFGS Algorithm
            direction = self.lbfgs(gradients)
        else:

            #Normal gradient descent is used when we have no previous parameters (the first step)
            direction = -gradients  

        #Updating pixel values of input image (the parameters) in the direction that minimizes loss
        with torch.no_grad():

            for parameter in self.params:
                
                #Updating each parameter in-place using PyTorch add_ in the direction of direction scaled by the learning rate
                parameter.add_(direction.view(parameter.size()), 
                               alpha=self.learning_rate)

        #Store current params and grads in prev variables for next step
        self.prev_params = parameters.clone()
        self.prev_grad = gradients.clone()

        return loss_at_step


    def zero_grad(self):
        """Zeroes the gradients at each optimization step. This is necessary to recompute distances accurately
        """

        for parameter in self.params:

            if parameter.grad is not None:

                #For each parameter, detach from computational graph and set gradient to zero
                parameter.grad.detach_()
                parameter.grad.zero_()


    def _get_flat_params(self):
        """Flattens parameters into a 1-D Vector

        Returns:
            pyTorch: Tensor
        """
        listed = []

        for parameter in self.params:
            listed.append(parameter.view(-1))

        concat = torch.cat(listed)
        return concat


    def _get_flat_grad(self):
        """Flattens gradients into a 1-D Vector

        Returns:
            pyTorch: Tensor
        """
        listed = []

        for parameter in self.params:
            listed.append(parameter.grad.view(-1))

        concat = torch.cat(listed)
        return concat


    def lbfgs(self, gradients):
        """The two-loop recursion implementation of LBFGS. When the paramter differences are large (the slope of loss function is steep),
           we have bigger scaling factors. We then subtract these factors from the gradients. Therefore, if we have a steep slope, we take
           smaller steps. If we have a flatter slope, we take larger steps.

        Args:
            gradients (Tensor): The current gradients at step n

        Returns:
            Tensor: The updated direction
        """
        #Clone so we do not alter in intermediate steps
        final_grads = gradients.clone()

        scaling_factors = []

        #Going backwards through the history of updates
        for i in range(len(self.parameter_differences) - 1, -1, -1):
            
            #Get current differences
            param_diff = self.parameter_differences[i]
            grad_diff = self.gradient_differences[i]
            scaling = self.scaling[i]

            #Get scaling factor
            scaling_factor_backward = (scaling * param_diff.dot(final_grads))
            scaling_factors.append(scaling_factor_backward)

            #Adjusting the final grads by subtracting scaling factors
            final_grads -= (scaling_factor_backward * grad_diff)

        #The intermediate grads after backward history pass
        intermediate = final_grads

        #Forward pass through the history of our parameter differences
        for i in range(len(self.parameter_differences)):

            #Get differences
            param_diff = self.parameter_differences[i]
            grad_diff = self.gradient_differences[i]
            scaling = self.scaling[i]

            scaling_factor_forward = (scaling * grad_diff.dot(intermediate))

            #Adding differences between parameter differences and difference of scaling factors
            intermediate += param_diff * (scaling_factors[i] - scaling_factor_forward)

        #Return the negative direction for minimization (we are trying to go in opposite direction of gradients)
        updated_gradient = -intermediate

        return updated_gradient


def pillow_transform(image_one_path:str, image_two_path:str):
    """Transforms an image path to a PIL image, then to a PyTorch tensor

    Args:
        image_one_path (str): Path to the first image
        image_two_path (str): Path to the second image

    Returns:
        tensor: PyTorch tensor versions of the image
    """
    image_one = Image.open(image_one_path)
    image_two = Image.open(image_two_path)

    tensor_transform = transforms.ToTensor()

    image_one = tensor_transform(image_one)
    image_two = tensor_transform(image_two)

    return image_one, image_two


def resize(style_image, content_image, image_size):
    """Resizes the two images to the same standard size (image_size x image_size). Will be a sqaure.

    Args:
        style_image (Pytorch Tensor): The style image
        content_image (Pytorch Tensor): The content image

    Returns:
        tensor: PyTorch tensor versions of the image resized to a square
    """

    transform = transforms.Resize((image_size, image_size))

    style_image = transform(style_image) 
    content_image = transform(content_image) 

    return style_image, content_image


def sharpen(style_image, content_image, image_sharpness=1.05):
    """Sharpen the image for better results after transfer

    Args:
        style_image (Pytorch Tensor): The style image
        content_image (Pytorch Tensor): The content image
        image_sharpness (float, optional): Sharpness factor. Greater than 1 is sharper. Defaults to 1.05.

    Returns:
        tensor: PyTorch tensor versions of the images sharpened
    """
    style_image = tranfunc.adjust_sharpness(style_image, image_sharpness)
    content_image = tranfunc.adjust_sharpness(content_image, image_sharpness)

    return style_image, content_image


def center_crop(style_image, content_image, crop_range=500):
    """Crops the center of the image.

    Args:
        style_image (Pytorch Tensor): The style image
        content_image (Pytorch Tensor): The content image
        crop_range (int, optional): The range from which to crop the image. Defaults to 500.

    Returns:
        tensor: PyTorch tensor versions of the image cropped
    """

    transform = transforms.CenterCrop((crop_range, crop_range)) 

    style_image = transform(style_image) 
    content_image = transform(content_image) 

    return style_image, content_image


def return_preprocessed_images(style_image_path, content_image_path, image_size, crop_range):
    """Applies all the preprocessing transformations and returns the adjusted images

    Args:
        style_image (Pytorch Tensor): The style image
        content_image (Pytorch Tensor): The content image
        image_size (int): The size of the image, will be a square
        crop_range (int): The range to crop the image

    Returns:
        tensor: PyTorch tensor versions of the images preprocessed for the network
    """
    style_image, content_image = pillow_transform(style_image_path, content_image_path)

    #If crop range is 0, ignore
    if crop_range != 0:
        style_image, content_image = center_crop(style_image, content_image, crop_range)

    style_image, content_image = resize(style_image, content_image, image_size)
    style_image, content_image = sharpen(style_image, content_image)

    #Adding batch dimension (necessary for PyTorch computations)
    style_image = style_image.unsqueeze(0)
    content_image = content_image.unsqueeze(0)

    return style_image, content_image


def display_images(style_image_path:str, content_image_path:str):
    """Function to display the two images before processing

    Args:
        style_image_path (str): File path to the style image
        content_image (str): File path to the content image
    """
    image_one = Image.open(style_image_path)
    image_two = Image.open(content_image_path)

    plt.title("Style Image before Preprocessing")
    plt.imshow(image_one)
    plt.pause(3)

    plt.title("Content Image before Preprocessing")
    plt.imshow(image_two)
    plt.pause(3)


def compute_gram_matrix(input):
    """Computes the gram matrix of an input image for the style computations

    Args:
        input (tensor): A PyTorch tensor representation of the image

    Returns:
        tensor: A returned gram matrix
    """
    #one is the batch dimension, which will be 1 (set with unsqueeze above), two is number of feature maps, 
    #three X four is each feature map size
    two = input.size()[1]
    three = input.size()[2]
    four = input.size()[3]

    #Features will be a feature_maps by feature_map_size tensor
    features = input.view(two, three * four)

    #Obtaining the gram matrix by multiplying features by its transpose
    gram_product = torch.mm(features, features.T)

    #Normalizing the gram matrix by dividing each element by the total number of elements in the matrix
    #This is necessary as large feature map sizes (three * four) will lead to large gram matrix values. These will weight
    #the early layers of the network more heavily (which we do not want)
    total_size = two * three * four
    final_gram = gram_product.div(total_size)

    return final_gram 


def trim_layers(convolutional_network):
    """Trim the excess layers of the VGG19 model by iterating through and finding the maximum index (last)
       loss layer

    Args:
        convolutional_network (PyTorch): The convolutional neural network

    Returns:
        int: The index of the last loss layer
    """
    #Trim layers that come after the last loss layer (content or style). The standard VGG-19 network has more layers
    #than what we want so we get the max index of a loss layer. Defaults to a non-value if it cannot find a loss
    last_loss_layer = max(
    (i for i, layer in enumerate(convolutional_network) if isinstance(layer, content_image_loss) \
        or isinstance(layer, style_image_loss)),

        default = -1000000
    )

    return last_loss_layer
    

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
    content_layers_default = set(["('7', Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))"])

    #The style layers are given in multiple layers, as style features can be high or low level
    style_layers_default = set(["('0', Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))",
                             "('2', Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))", 
                            "('5', Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))", 
                            "('7', Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))", 
                            "('10', Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))",
                            "('12', Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))"])
    
    #Creating normalization module and initializing the model with a normalization layer
    normalization = normalize_images(network_mean, network_std)
    model = nn.Sequential(normalization)

    content_loss_list = []
    style_loss_list = []

    #Iterate through each layer of the convolutional network
    for layer in convolutional_network.named_children():

        layer_type = type(layer[-1])
        if layer_type in (nn.Conv2d, nn.ReLU, nn.MaxPool2d, nn.BatchNorm2d):
            name = str(layer)

            #inplace ReLU recommended for neural style transfer
            if layer_type == nn.ReLU:
                layer = nn.ReLU(inplace=False)

        #Adding the current layer to the model with the appropriate name
        if type(layer) == tuple:
            if isinstance(layer[-1], nn.Conv2d):
                layer_add = layer[-1]
        else:
            layer_add = layer

        #Adding the layer to the model with appropriate name
        model.add_module(name, layer_add)

        #Add content loss after the specified content layers
        if name in content_layers_default:

            #The feature maps of the content image we want to match
            content_target_filters = model(image_one).detach()

            #Create the content loss layer
            content_loss = content_image_loss(content_target_filters)

            #Adding content loss layer to the model
            model.add_module(f'content_loss_{layer[0]}', content_loss)

            #Append the content loss to the list
            content_loss_list.append(content_loss)

        #Add style loss after the specified style layers
        if name in style_layers_default:
            #The feature maps of the style image, using the Gram matrix
            style_target_filters = model(image_two).detach()

            #Create the style loss layer
            style_loss = style_image_loss(style_target_filters)

            #Adding style loss layer to the model
            model.add_module(f'style_loss_{layer[0]}', style_loss)

            #Append the style loss to the list
            style_loss_list.append(style_loss)

    index = trim_layers(model)
    #Trim layers that come after the last loss layer through indexing
    model = model[:index + 1]

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


def get_manual_input_image_optimizer(input_image:torch.tensor, learning_rate):
    """Limited Memory BFGS is an optimization algorithm. Computes the gradient of the loss function
       and asjusts the parameters (pixel values in this case) in the direction of steepest descent
       of the loss function. This retrieves the manual implementation version of the algorithm.

    Args:
        input_image (torch.tensor): The input image as a tensor

    Returns:
        optimizer: The appropriate optimizer
    """
    optm = ManualLBFGS([input_image], learning_rate=learning_rate)

    return optm


def neural_style_transfer_algorithm(cnn, normalization_mean, normalization_std,
                       content_image, style_image, input_image, num_steps=300,
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

    #Retrieving the baseline losses
    model, content_losses, style_losses = style_content_loss_layers(cnn, normalization_mean, normalization_std,
                               content_image, style_image
    )

    #Input image needs grad to compute loss and adjust pixel values
    input_image.requires_grad_(True)

    #We are not updating model weights so we set to eval() 
    model.eval()
    model.requires_grad_(False)

    #Getting the optimizer to adjust pixel values for the input image
    optimizer = get_manual_input_image_optimizer(input_image, learning_rate=0.0001)

    #Define run as an array due to Python scope. Arrays are mutable, whereas integers are not, within the 
    #inner function 'optimize'
    num_steps_array = [0]

    while num_steps_array[0] <= num_steps:

        def optimize():
            """The function that returns the loss for the optimizer. It computes the weighted loss
               of both and calls backward() on this loss. This computes the gradients needed by the optimizer
               to change the input image. 

            Returns:
                float: The loss
            """
            #Ensure pixel values are between 0 and 1
            input_image.data.clamp_(0, 1)

            #Zero the gradients at each step
            optimizer.zero_grad()

            model(input_image)

            style, content = 0, 0

            for sl in style_losses:
                style += sl.loss
            style = style * style_weight

            #Only one layer used in content loss
            content = content_weight * content_losses[0].loss

            #total loss is the weighted sum of the two separate losses
            total_loss = style + content

            #Computing all the gradients with backward()
            total_loss.backward()

            #Increment the step
            num_steps_array[0] += 1


            if num_steps_array[0] % 20 == 0:

                print(f"Step {num_steps_array[0]}: Style Loss: \
                      {style.item():.4f}, Content Loss: {content.item():.4f}")
                
            return total_loss

        optimizer.step(optimize)

    #Ensuring final image has valid pixel values between 0 and 1 (sanity check)
    with torch.no_grad():

        input_image.clamp_(0, 1)

    return input_image


if __name__ == '__main__':

    #Set to eval because layers differ in behavior during evaluation and training
    #VGG19 is split into two components: 'features' and 'classifier'. Features contains the 
    #convolutional and pooling layers, which is what we want for neural-style transfer. Also,
    #we are not altering the model's default weights so we set the mode to eval() (We are not training kernels)
    convolutional_network = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()

    #Normalizing images based on the vgg19 network. The vgg19 network is trained on images
    #with channels normalized by the following values
    network_image_mean = torch.tensor([0.485, 0.456, 0.406])
    network_image_std = torch.tensor([0.229, 0.224, 0.225])

    #Setting the standard image size for content, style, and output images (larger sizes will take longer)
    image_size_global = 128
    image_one_path = "Images\MonaLisa.jfif"
    image_two_path = "Images\Acrisure.jfif"
    white_noise = "Images\WhiteNoise.jpg"

    #Displaying images
    display_images(style_image_path=image_one_path, content_image_path=image_two_path)

    style_image, content_image = return_preprocessed_images(style_image_path=image_one_path, content_image_path=image_two_path,
                                                            image_size=image_size_global, crop_range=0)
    input_image = content_image.clone()

    #Setting the weights for the loss
    content_weight = 1
    style_weight = 1000000

    #Obtaining output image
    output_image = neural_style_transfer_algorithm(convolutional_network, network_image_mean, network_image_std,
                                                   content_image, style_image, input_image, num_steps=360, content_weight=content_weight,
                                                   style_weight=style_weight)

    plt.figure()
    plt.title(f"Ouput with Content Weight: {content_weight} and Style Weight: {style_weight}")

    #Reformatting output image for display (image is in PyTorch format, we re-adjust channels and get rid of batch dimension)
    output_image = output_image.squeeze(0).detach().numpy()  
    output_image = output_image.transpose(1, 2, 0)  
    plt.imshow(output_image)
    plt.pause(100)








