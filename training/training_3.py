import numpy as np
import torch
import matplotlib.pyplot as plt
from IPython import display
import wandb

from helpers import gan_loss_func

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_gan(config, g, d, train_loader, p_name='tmp'):
    # Initialize wandb run
    wandb.init(project=p_name, config=config)

    # Optimizers for generator and discriminator
    g_opt = torch.optim.Adam(g.parameters(), lr=config["lr_g"], betas=(0.5, 0.999))
    d_opt = torch.optim.Adam(d.parameters(), lr=config["lr_d"], betas=(0.5, 0.999))

    # set loss function
    gan_loss = gan_loss_func(config)
    
    # Create a figure
    plt.figure(figsize=(20, 10))
    subplots = [plt.subplot(2, 6, k+1) for k in range(12)]

    # perform training
    for epoch in range(config['epochs']):
        for minibatch_no, (x, target) in enumerate(train_loader):
            x_real = x.to(device) * 2 - 1  # scale to (-1, 1) range
            z = torch.randn(x.shape[0], 100).to(device)
            x_fake = g(z)
            
            # Update discriminator 
            d.zero_grad()
            d_loss = gan_loss.discriminator(d, x_real, x_fake)
            d_loss.backward()
            d_opt.step()

            # Update generator
            g.zero_grad()
            g_loss = gan_loss.generator(d, x_real, x_fake)
            g_loss.backward()
            g_opt.step()

            assert(not np.isnan(d_loss.item()))
            #Plot results every 100 minibatches
            if minibatch_no % 100 == 0:
                title = 'Epoch {e} - minibatch {n}/{d}'.format(e=epoch+1, n=minibatch_no, d=len(train_loader))
                visualize_train(config, g, d, x_real, x_fake, subplots, d_loss, title)
    wandb.finish()


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
