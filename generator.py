import config
from helpers import normal_init

class generator(nn.Module):
  # initializers
  def __init__(self):
    super(generator, self).__init__()
    ######################################
    # Unet generator encoder             #
    ######################################
    self.enc1 = nn.Sequential(
        nn.Conv2d(3, 64, 4, 2, 1, padding_mode='reflect'),
        nn.LeakyReLU(0.2)
    )
    self.enc2 = nn.Sequential(
        nn.Conv2d(64, 128, 4, 2, 1, padding_mode='reflect'),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(0.2)
    )
    self.enc3 = nn.Sequential(
        nn.Conv2d(128, 256, 4, 2, 1, padding_mode='reflect'),
        nn.BatchNorm2d(256),
        nn.LeakyReLU(0.2)
    )
    self.enc4 = nn.Sequential(
        nn.Conv2d(256, 512, 4, 2, 1, padding_mode='reflect'),
        nn.BatchNorm2d(512),
        nn.LeakyReLU(0.2)
    )
    self.enc5 = nn.Sequential(
        nn.Conv2d(512, 512, 4, 2, 1, padding_mode='reflect'),
        nn.BatchNorm2d(512),
        nn.LeakyReLU(0.2)
    )
    self.enc6 = nn.Sequential(
        nn.Conv2d(512, 512, 4, 2, 1, padding_mode='reflect'),
        nn.BatchNorm2d(512),
        nn.LeakyReLU(0.2)
    )
    self.enc7 = nn.Sequential(
        nn.Conv2d(512, 512, 4, 2, 1, padding_mode='reflect'),
        nn.BatchNorm2d(512),
        nn.LeakyReLU(0.2)
    )
    self.enc8 = nn.Sequential(
        nn.Conv2d(512, 512, 4, 2, 1, padding_mode='reflect'),
        nn.BatchNorm2d(512),
        nn.LeakyReLU(0.2)
    )
    ######################################
    # Unet generator decoder             #
    ######################################
    self.dec1 = nn.Sequential(
        nn.ConvTranspose2d(512, 512, 4, 2, 1),
        nn.BatchNorm2d(512),
        nn.Dropout(0.5),
        nn.ReLU()
    )
    self.dec2 = nn.Sequential(
        nn.ConvTranspose2d(1024, 512, 4, 2, 1),
        nn.BatchNorm2d(512),
        nn.Dropout(0.5),
        nn.ReLU()
    )
    self.dec3 = nn.Sequential(
        nn.ConvTranspose2d(1024, 512, 4, 2, 1),
        nn.BatchNorm2d(512),
        nn.Dropout(0.5),
        nn.ReLU()
    )
    self.dec4 = nn.Sequential(
        nn.ConvTranspose2d(1024, 512, 4, 2, 1),
        nn.BatchNorm2d(512),
        nn.ReLU()
    )
    self.dec5 = nn.Sequential(
        nn.ConvTranspose2d(1024, 256, 4, 2, 1),
        nn.BatchNorm2d(256),
        nn.ReLU()
    )
    self.dec6 = nn.Sequential(
        nn.ConvTranspose2d(512, 128, 4, 2, 1),
        nn.BatchNorm2d(128),
        nn.ReLU()
    )
    self.dec7 = nn.Sequential(
        nn.ConvTranspose2d(256, 64, 4, 2, 1),
        nn.BatchNorm2d(64),
        nn.ReLU()
    )
    self.dec8 = nn.Sequential(
        nn.ConvTranspose2d(128, 3, 4, 2, 1),
        nn.Tanh()
    )

  # weight_init
  def weight_init(self, mean, std):
    for m in self._modules:
      normal_init(self._modules[m], mean, std)

  # forward method
  def forward(self, input):
    # encoding
    e1 = self.enc1(input)
    e2 = self.enc2(e1)
    e3 = self.enc3(e2)
    e4 = self.enc4(e3)
    e5 = self.enc5(e4)
    e6 = self.enc6(e5)
    e7 = self.enc7(e6)
    bottleneck = self.enc8(e7)

    # decoding
    d1 = self.dec1(bottleneck)
    d2 = self.dec2(torch.cat([d1, e7], 1))
    d3 = self.dec3(torch.cat([d2, e6], 1))
    d4 = self.dec4(torch.cat([d3, e5], 1))
    d5 = self.dec5(torch.cat([d4, e4], 1))
    d6 = self.dec6(torch.cat([d5, e3], 1))
    d7 = self.dec7(torch.cat([d6, e2], 1))
    output = self.dec8(torch.cat([d7, e1], 1))

    return output