import torch
import torch.nn as nn

""" 
Information about architecture config:
Tuple is structured by (kernel_size, filters, stride, padding) 
"M" is simply maxpooling with stride 2x2 and kernel 2x2
List is structured by tuples and lastly int with number of repeats
"""

architecture_config = [
    #Tuple: (kernel_size, number of filters, strides, padding)
    (7, 64, 2, 3),
    #"M" = Max Pool Layer
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    #List: [(tuple), (tuple), how many times to repeat]
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
    #Doesnt include fc layers
]

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)
        
    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))
    
class YoloV1(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super(YoloV1, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self._create_darknet(self.architecture)
        self.fcs = self._create_fcs(**kwargs)
        
    def forward(self, x):
        """
        The function takes in an input `x` and passes it through the `darknet` model.
        The output of the `darknet` model is then flattened and passed through the `fcs`
        model. The output of the `fcs` model is then returned.
        
        @param x the input to the model
        @return The output of the last layer of the network.
        """
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1))
    
    def _create_darknet(self, architecture):
        """
        It takes in a list of tuples, strings, and lists, and returns a sequential
        list of CNNBlocks and MaxPool2d layers
        
        @param architecture A list of tuples, strings, and lists.
        @return A sequential list of CNNBlocks and MaxPool2d layers.
        """
        layers = []
        in_channels = self.in_channels
        
        for x in architecture:
            if type(x) == tuple:
                layers += [CNNBlock(in_channels, x[1], kernel_size=x[0], stride=x[2], padding=x[3])]
                in_channels = x[1]
            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif type(x) == list:
                conv1 = x[0] #Tuple
                conv2 = x[1] #Tuple
                repeats = x[2] #Int
                
                for _ in range(repeats):
                    layers += [CNNBlock(in_channels, conv1[1], kernel_size=conv1[0], stride=conv1[2], padding=conv1[3])]
                    layers += [CNNBlock(conv1[1], conv2[1], kernel_size=conv2[0], stride=conv2[2], padding=conv2[3])]
                    in_channels = conv2[1]
                    
        return nn.Sequential(*layers)
    
    def _create_fcs(self, split_size, num_boxes, num_classes):
        """
        The function takes in the split size, number of boxes and number of classes
        and returns a sequential model with a flatten layer, two linear layers, a
        dropout layer and a leaky relu layer
        
        @param split_size The size of the image after it's been split into grids.
        @param num_boxes number of boxes per cell
        @param num_classes number of classes in the dataset
        @return The output of the fc_sequential is the output of the last layer.
        """
        S, B, C = split_size, num_boxes, num_classes
        fc_sequential = nn.Sequential(
                        nn.Flatten(), 
                        nn.Linear(S * S * 1024, 4096), 
                        nn.Dropout(0.5), 
                        nn.LeakyReLU(0.1), 
                        nn.Linear(4096, S * S * (C + B * 5))
                        )
        #Original paper uses nn.Linear(1024 * S * S, 4096) not 496. Also the last layer will be reshaped to (S, S, 13) where C+B*5 = 13
        return fc_sequential