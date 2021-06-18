import numpy as np
import torch
import matplotlib.pyplot as plt
from IPython import display
import wandb

from helpers import gan_im_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_cycle_gan(config, g_h2z, g_z2h, d_h, d_z, zebra_loader, horse_loader, p_name='tmp'):
    """Training function for the Cycle GAN network
    Inputs
        config          :   A config dict that is sent to weight and biases, 
                            and used to set different training hyperparameters
                                'lr_g'          :   Learning rate for the 
                                                    generator optimizer
                                'lr_d'          :   Learning rate for the 
                                                    discriminator optimizer
                                'epochs'        :   Number of epochs to train
                                'img_loss'      :   Specify how to compute the 
                                                    image loss ('l1' or 'l2')
                                'g_loss_weight' :   Specify the weighting of
                                                    the generator losses as a list 
                                                    of ints: [fool, cycle, identity]
        g_h2z           :   A generator nn.Module that converts horses to zebras
        g_z2h           :   A generator nn.Module that converts zebras to horses
        d_h             :   A discriminator nn.Module that can discriminate horses
        d_z             :   A discriminator nn.Module that can discriminate zebras
        zebra_loader    :   A Dataloader of the zebra training data
        horse_loader    :   A Dataloader of the horse training data
        p_name          :   A string, determining the name of the project on wandb
    """
    # Define loss functions as LSGAN
    def real_loss(x):
        return torch.mean((x - 1)**2)

    def fake_loss(x):
        return torch.mean(x**2)
    
    # Define image loss as L2 loss
    im_loss = gan_im_loss(config)
    
    # Initialize wandb run
    # wandb.init(project=p_name, config=config)

    # Optimizers for generators and discriminators
    g_param = list(g_h2z.parameters()) + list(g_z2h.parameters())
    d_param = list(d_h.parameters()) + list(d_z.parameters())
    g_opt = torch.optim.Adam(g_param, lr=config["lr_g"], betas=(0.5, 0.999))
    d_opt = torch.optim.Adam(d_param, lr=config["lr_d"], betas=(0.5, 0.999))

    # Converte loaders to iterators
    data_zebra = iter(zebra_loader)
    data_horse = iter(horse_loader)

    # perform training
    for epoch in range(config['epochs']):
        for i in range(len(data_zebra)):
            # Get batch
            x_zebra = next(data_zebra).to(device)
            x_horse = next(data_horse).to(device)

            # Generate fake images
            x_zebra_fake = g_h2z(x_horse)
            x_horse_fake = g_z2h(x_zebra)

            # Generate recreational images
            x_zebra_rec = g_h2z(x_horse_fake)
            x_horse_rec = g_z2h(x_zebra_fake)
            
            # Update discriminator 
            d_opt.zero_grad()
            d_loss = real_loss(d_h(x_horse))
            d_loss += fake_loss(d_h(x_horse_fake.detach()))
            d_loss += real_loss(d_z(x_zebra))
            d_loss += fake_loss(d_z(x_zebra_fake.detach()))
            d_loss.backward()
            d_opt.step()

            # Update generator
            g_opt.zero_grad()
            g_loss_fool = real_loss(d_h(x_horse_fake))
            g_loss_fool += real_loss(d_z(x_zebra_fake))
            g_loss_fool *= config['g_loss_weight'][0]
            g_loss_cycle = im_loss(x_horse, x_horse_rec)
            g_loss_cycle += im_loss(x_zebra, x_zebra_rec)
            g_loss_cycle *= config['g_loss_weight'][1]
            g_loss_iden = im_loss(g_h2z(x_zebra), x_zebra)
            g_loss_iden += im_loss(g_z2h(x_horse), x_horse)
            g_loss_iden *= config['g_loss_weight'][2]
            g_loss = g_loss_fool + g_loss_cycle + g_loss_iden
            g_loss.backward()
            g_opt.step()

            assert(not np.isnan(d_loss.item()))
            #Plot results every 100 minibatches
            if i % 100 == 0:
                continue
                # title = 'Epoch {e} - minibatch {n}/{d}'.format(e=epoch+1, n=i, d=len(train_loader))
                # visualize_train(config, g, d, x_real, x_fake, subplots, d_loss, title)
    # wandb.finish()


def visualize_train(config, g, d, x_real, x_fake, subplots, d_loss, title):
    with torch.no_grad():
        P = torch.sigmoid(d(x_fake))
        for k in range(11):
            x_fake_k = x_fake[k].cpu().squeeze()/2+.5
            subplots[k].imshow(x_fake_k, cmap='gray')
            subplots[k].set_title('d(x)=%.2f' % P[k])
            subplots[k].axis('off')
        z = torch.randn(config['batch_size'], 100).to(device)
        H1 = torch.sigmoid(d(g(z))).cpu()
        H2 = torch.sigmoid(d(x_real)).cpu()
        plot_min = min(H1.min(), H2.min()).item()
        plot_max = max(H1.max(), H2.max()).item()
        subplots[-1].cla()
        subplots[-1].hist(H1.squeeze(), label='fake', range=(plot_min, plot_max), alpha=0.5)
        subplots[-1].hist(H2.squeeze(), label='real', range=(plot_min, plot_max), alpha=0.5)
        subplots[-1].legend()
        subplots[-1].set_xlabel('Probability of being real')
        subplots[-1].set_title('Discriminator loss: %.2f' % d_loss.item())
        
        plt.gcf().suptitle(title, fontsize=20)
        plt.savefig('log_image.png', transparent=True, bbox_inches='tight')
        display.display(plt.gcf())
        display.clear_output(wait=True)
        wandb.log({"Train Visualization": wandb.Image("log_image.png")})
