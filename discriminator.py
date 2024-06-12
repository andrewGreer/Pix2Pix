import config

class discriminator(nn.Module):
  # initializers
  def __init__(self):
    super(discriminator, self).__init__()

    self.disc1 = nn.Sequential(
        nn.Conv2d(6, 64, 4, 2, 1, padding_mode='reflect'),
        nn.LeakyReLU(0.2)
    )
    self.disc2 = nn.Sequential(
        nn.Conv2d(64, 128, 4, 2, 1, padding_mode='reflect'),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(0.2)
    )
    self.disc3 = nn.Sequential(
        nn.Conv2d(128, 256, 4, 2, 1, padding_mode='reflect'),
        nn.BatchNorm2d(256),
        nn.LeakyReLU(0.2)
    )
    self.disc4 = nn.Sequential(
        nn.Conv2d(256, 512, 4, 2, 1, padding_mode='reflect'),
        nn.BatchNorm2d(512),
        nn.LeakyReLU(0.2)
    )
    self.disc5 = nn.Sequential(
        nn.Conv2d(512, 1, 4, 1, 0, padding_mode='reflect'),
        nn.Sigmoid()
    )

  # weight_init
  def weight_init(self, mean, std):
    for m in self._modules:
      normal_init(self._modules[m], mean, std)

  # forward method
  def forward(self, input):
    x = self.disc1(input)
    x = self.disc2(x)
    x = self.disc3(x)
    x = self.disc4(x)
    x = self.disc5(x)

    return x
