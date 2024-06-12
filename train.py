import config
import helpers

L1_loss = nn.L1Loss().cuda()
MSE_loss = nn.MSELoss().cuda()
BCE_loss = nn.BCELoss().cuda()

def train(G, D, train_loader=None, num_epochs=20, only_L1=False, only_L2=False, both_L1_cGAN=False, both_L2_cGAN=False, flip=False):
  hist_D_losses = []
  hist_G_losses = []
  hist_G_cGAN_losses = []     # cGAN only
  hist_G_L1_losses = []       # L1 only
  hist_G_L2_losses = []       # L2 only
  hist_G_L1_cGAN_losses = []  # L1 + cGAN
  hist_G_L2_cGAN_losses = []  # L2 + cGAN

  # Adam optimizers
  G_optimizer = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
  D_optimizer = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

  print('training start!')
  start_time = time.time()
  for epoch in range(num_epochs):
    print('Start training epoch %d' % (epoch + 1))
    D_losses = []
    G_losses = []
    epoch_start_time = time.time()
    num_iter = 0

    for x_ in train_loader:

      if flip:
        y_ = x_[:, :, :, 0:img_size]
        x_ = x_[:, :, :, img_size:]
      else:
        y_ = x_[:, :, :, img_size:]
        x_ = x_[:, :, :, 0:img_size]

      x_, y_ = x_.cuda(), y_.cuda()
      ###########################################################################
      #                         Train the discriminator                         #
      ###########################################################################
      D.zero_grad()

      real_input = torch.cat([x_, y_], 1)
      D_real = D(real_input).squeeze()
      D_real_loss = BCE_loss(D_real, torch.ones(D_real.size()).cuda())

      fake_input = torch.cat([x_, G(x_)], 1)
      D_fake = D(fake_input).squeeze()
      D_fake_loss = BCE_loss(D_fake, torch.zeros(D_fake.size()).cuda())

      D_train_loss = (D_real_loss + D_fake_loss) / 2
      D_train_loss.backward()
      D_optimizer.step()
      loss_D = D_train_loss.detach().item()

      ###########################################################################
      #                              Train Generator                            #
      ###########################################################################

      G.zero_grad()

      G_result = G(x_)
      fake_input = torch.cat([x_, G_result], 1)
      D_fake = D(fake_input).squeeze()

      if only_L1:
        G_train_loss = L1_loss(G_result, y_)
        hist_G_losses.append(L1_loss(G_result, y_).detach().item())
        hist_G_L1_losses.append(L1_loss(G_result, y_).detach().item())
      elif only_L2:
        G_train_loss = MSE_loss(G_result, y_)
        hist_G_losses.append(MSE_loss(G_result, y_).detach().item())
        hist_G_L2_losses.append(MSE_loss(G_result, y_).detach().item())
      elif both_L1_cGAN:
        G_train_loss = BCE_loss(D_fake, torch.ones(D_fake.size()).cuda()) + 100 * L1_loss(G_result, y_)
        hist_G_losses.append((BCE_loss(D_fake, torch.ones(D_fake.size()).cuda()) + 100 * L1_loss(G_result, y_)).detach().item())
        hist_G_L1_losses.append(L1_loss(G_result, y_).detach().item())
        hist_G_cGAN_losses.append(BCE_loss(D_fake, torch.ones(D_fake.size()).cuda()).detach().item())
        hist_G_L1_cGAN_losses.append((BCE_loss(D_fake, torch.ones(D_fake.size()).cuda()) + 100 * L1_loss(G_result, y_)).detach().item())
      elif both_L2_cGAN:
        G_train_loss = BCE_loss(D_fake, torch.ones(D_fake.size()).cuda()) + 100 * MSE_loss(G_result, y_)
        hist_G_losses.append((BCE_loss(D_fake, torch.ones(D_fake.size()).cuda()) + 100 * MSE_loss(G_result, y_)).detach().item())
        hist_G_L2_losses.append(MSE_loss(G_result, y_).detach().item())
        hist_G_cGAN_losses.append(BCE_loss(D_fake, torch.ones(D_fake.size()).cuda()).detach().item())
        hist_G_L2_cGAN_losses.append((BCE_loss(D_fake, torch.ones(D_fake.size()).cuda()) + 100 * MSE_loss(G_result, y_)).detach().item())
      else: # cGAN
        G_train_loss = BCE_loss(D_fake, torch.ones(D_fake.size()).cuda())
        hist_G_losses.append(BCE_loss(D_fake, torch.ones(D_fake.size()).cuda()).detach().item())
        hist_G_cGAN_losses.append(BCE_loss(D_fake, torch.ones(D_fake.size()).cuda()).detach().item())


      G_train_loss.backward()
      G_optimizer.step()
      loss_G = G_train_loss.detach().item()

      D_losses.append(loss_D)
      hist_D_losses.append(loss_D)
      G_losses.append(loss_G)
      num_iter += 1

    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time

    print('[%d/%d] - using time: %.2f seconds' % ((epoch + 1), num_epochs, per_epoch_ptime))
    print('loss of discriminator D: %.3f' % (torch.mean(torch.FloatTensor(D_losses))))
    print('loss of generator G: %.3f' % (torch.mean(torch.FloatTensor(G_losses))))

    if (epoch + 1) % num_epochs == 0:
      with torch.no_grad():
        if only_L1:
          show_result(G, fixed_x_, fixed_y_, (epoch+1), only_L1=True)
        elif only_L2:
          show_result(G, fixed_x_, fixed_y_, (epoch+1), only_L2=True)
        elif both_L1_cGAN:
          show_result(G, fixed_x_, fixed_y_, (epoch+1), both_L1_cGAN=True)
        elif both_L2_cGAN:
          show_result(G, fixed_x_, fixed_y_, (epoch+1), both_L2_cGAN=True)
        else:
          show_result(G, fixed_x_, fixed_y_, (epoch+1))


  end_time = time.time()
  total_ptime = end_time - start_time

  return hist_D_losses, hist_G_losses, hist_G_L1_losses, hist_G_L2_losses, hist_G_L1_cGAN_losses, hist_G_L2_cGAN_losses, hist_G_cGAN_losses