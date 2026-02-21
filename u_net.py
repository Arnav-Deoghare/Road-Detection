import torch

def double_conv(in_channels, out_channels):
    conv=torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        torch.nn.ReLU(inplace=True),
        torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        torch.nn.ReLU(inplace=True))
    return conv

def MaxPool2d(kernel_size):
    return torch.nn.MaxPool2d(kernel_size=kernel_size, stride=kernel_size)

def ConvTranspose2d(in_channels, out_channels):
    return torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
uhuhu
class UNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = double_conv(3, 64)
        self.enc2 = double_conv(64, 128)
        # self.enc3 = double_conv(128, 256)
        # self.enc4 = double_conv(256, 512)
        # self.enc5 = double_conv(512, 1024)
        self.pool = MaxPool2d(2)


        self.up1 = ConvTranspose2d(128,128)
        self.dec1 = double_conv(256,128)

        self.up2 = ConvTranspose2d(128,64)
        self.dec2 = double_conv(128,64)

        self.final = torch.nn.Conv2d(64, 1, kernel_size=1)


    def forward(self, x):
        x1 = self.enc1(x)
        x = self.pool(x1)

        x2 = self.enc2(x)
        x =  self.pool(x2)

        # x3 = self.enc3(x)
        # x = self.pool(x3)   

        # x4 = self.enc4(x)
        # x = self.pool(x4)   

        # x5 = self.enc5(x)        
        x = self.up1(x)
        x = torch.cat([x, x2], dim=1)
        x = self.dec1(x)

        x = self.up2(x)
        x = torch.cat([x, x1], dim=1)
        x = self.dec2(x)
        x = self.final(x)
    
        x = torch.sigmoid(x)

        return x


# model = UNet()
# x = torch.randn(1, 3, 256, 256)
# out = model(x)
# print(out.shape)
