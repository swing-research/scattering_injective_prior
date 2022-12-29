import tensorflow as tf
from tensorflow.keras import layers


class Conv_block(layers.Layer):
    def __init__(self , num_filters):
        super(Conv_block, self).__init__()

        self.conv1 = layers.Conv2D(num_filters, (3, 3), padding="same", use_bias=False)
        self.conv2 = layers.Conv2D(num_filters, (3, 3), padding="same", use_bias=False)
        self.act1 = layers.Activation("relu")
        self.act2 = layers.Activation("relu")

    def call(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        
        x = self.conv2(x)
        x = self.act2(x)
        return x
    

class Unet(layers.Layer):
    def __init__(self , output_channels):
        super(Unet, self).__init__()
        
        self.num_filters = [32, 64]
        
        self.conv_blocks1 = [Conv_block(num_filters = f) 
                            for f in self.num_filters]
        
        self.conv_blocks2 = [Conv_block(num_filters = f) 
                            for f in self.num_filters[::-1]]
            
        self.conv_block_bridge = Conv_block(self.num_filters[-1])
        self.maxpool = layers.MaxPool2D((2, 2))
        self.upsample = layers.UpSampling2D((2, 2))
        self.concat = layers.Concatenate()
        self.conv = layers.Conv2D(output_channels, (1, 1), padding="same" , use_bias=False)
        self.act = layers.Activation("sigmoid")
        

    def call(self, x , training = True):
        
        skip_x = []
        
        ## Encoder
        for i in range(len(self.num_filters)):
            x = self.conv_blocks1[i](x)
            skip_x.append(x)
            x = self.maxpool(x)
        
        ## Bridge
        x = self.conv_block_bridge(x)

        skip_x = skip_x[::-1]
        ## Decoder
        for i in range(len(self.num_filters)):
            x = self.upsample(x)
            xs = skip_x[i]
            x = self.concat([x, xs])
            x = self.conv_blocks2[i](x)

    
        ## Output
        x = self.conv(x)
        x = self.act(x)
        return x
