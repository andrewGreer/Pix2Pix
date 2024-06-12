import config

def normal_init(m, mean, std):
    # Initialize model parameter with given mean and std
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

L1_images = []
L2_images = []
cGAN_images = []
L1_cGAN_images = []
L2_cGAN_images = []

def compose_output():
  # Visualization
  fig, axs = plt.subplots(4, 5, figsize=(15, 12))  # 4 rows, 5 columns
  for i in range(4):  # Assuming you have at least 4 sets of images
      axs[i, 0].imshow(transforms.ToPILImage()(gen_images_L1[i][0]))
      axs[i, 0].set_title("Input Image")
      axs[i, 1].imshow(transforms.ToPILImage()(gen_images_L1[i][1]))
      axs[i, 1].set_title("Ground Truth")
      axs[i, 2].imshow(transforms.ToPILImage()(gen_images_L1[i][2]))
      axs[i, 2].set_title("Output L1")
      axs[i, 3].imshow

# Helper function for showing result.
def process_image(img):
  return (img.cpu().data.numpy().transpose(1, 2, 0) + 1) / 2

def show_result(G, x_, y_, num_epoch, only_L1=False, only_L2=False, both_L1_cGAN=False, both_L2_cGAN=False):
  predict_images = G(x_)

  fig, ax = plt.subplots(x_.size()[0], 3, figsize=(6,10))

  for i in range(x_.size()[0]):
    ax[i, 0].get_xaxis().set_visible(False)
    ax[i, 0].get_yaxis().set_visible(False)
    ax[i, 1].get_xaxis().set_visible(False)
    ax[i, 1].get_yaxis().set_visible(False)
    ax[i, 2].get_xaxis().set_visible(False)
    ax[i, 2].get_yaxis().set_visible(False)

    ax[i, 0].cla()
    img1 = (process_image(x_[i]) * 255).astype(np.uint8)
    ax[i, 0].imshow(img1)
    ax[i, 1].cla()
    img2 = (process_image(predict_images[i]) * 255).astype(np.uint8)
    ax[i, 1].imshow(img2)
    ax[i, 2].cla()
    img3 = (process_image(y_[i]) * 255).astype(np.uint8)
    ax[i, 2].imshow(img3)

    image_dict = {
        'input': img1,
        'output': img2,
        'truth': img3
    }

    if only_L1:
      L1_images.append(image_dict)
    elif only_L2:
      L2_images.append(image_dict)
    elif both_L1_cGAN:
      L1_cGAN_images.append(image_dict)
    elif both_L2_cGAN:
      L2_cGAN_images.append(image_dict)
    else:
      cGAN_images.append(image_dict)

  plt.subplots_adjust(wspace=0.0, hspace=0.0)

  plt.tight_layout(pad=0.1, h_pad=0.1, w_pad=0.1)
  label_epoch = 'Epoch {0}'.format(num_epoch)
  fig.text(0.5, 0, label_epoch, ha='center')
  label_input = 'Input'
  fig.text(0.18, 1, label_input, ha='center')
  label_output = 'Output'
  fig.text(0.5, 1, label_output, ha='center')
  label_truth = 'Ground truth'
  fig.text(0.81, 1, label_truth, ha='center')

  plt.show()

# Helper function for counting number of trainable parameters.
def count_params(model):
  '''
  Counts the number of trainable parameters in PyTorch.
  Args:
      model: PyTorch model.
  Returns:
      num_params: int, number of trainable parameters.
  '''
  num_params = sum([item.numel() for item in model.parameters() if item.requires_grad])
  return num_params