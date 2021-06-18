from os import path
import numpy as np
import torch
import matplotlib.pyplot as plt
from IPython import display
import wandb

from helpers import gan_im_loss, ImageBuffer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
save_folder = 'saved_states'

# Define loss functions as LSGAN
def real_loss(x):
    return torch.mean((x - 1)**2)

def fake_loss(x):
    return torch.mean(x**2)


def train_cycle_gan(config, g_h2z, g_z2h, d_h, d_z, zebra_loader, horse_loader, p_name='tmp', plotting=False):
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
    # Define image losses
    im_loss_1, im_loss_2  = gan_im_loss(config)

    # Initialize wandb run
    wandb.init(project=p_name, config=config)

    # Optimizers for generators and discriminators
    g_param = list(g_h2z.parameters()) + list(g_z2h.parameters())
    d_param = list(d_h.parameters()) + list(d_z.parameters())
    g_opt = torch.optim.Adam(g_param, lr=config["lr_g"], betas=(0.5, 0.999))
    d_opt = torch.optim.Adam(d_param, lr=config["lr_d"], betas=(0.5, 0.999))

    if "buffer_size" in config and config["buffer_size"]:
        fake_h_buffer = ImageBuffer(config["buffer_size"])
        fake_z_buffer = ImageBuffer(config["buffer_size"])

    # perform training
    for epoch in range(config['epochs']):
        print(f"Epoch {epoch+1}/{config['epochs']}")

        # Converte loaders to iterators
        data_zebra = iter(zebra_loader)
        data_horse = iter(horse_loader)

        # Define logging dict for wandb
        logging = {
            'd_loss': 0,
            'g_loss': 0,
            'g_loss_fool': 0,
            'g_loss_cycle': 0,
            'g_loss_iden': 0,
        }

        # Go over all batches
        for i in range(len(data_zebra)):
            # Get zebras
            x_zebra = next(data_zebra).to(device) * 2 - 1
            
            # Get horses
            # Using hack to keep getting horse images, even though there are less of them
            # https://discuss.pytorch.org/t/infinite-dataloader/17903/10
            try: 
                x_horse = next(data_horse).to(device) * 2 - 1
            except StopIteration:
                data_horse = iter(horse_loader)
                x_horse = next(data_horse).to(device) * 2 - 1

            # Generate fake images
            x_zebra_fake = g_h2z(x_horse)
            x_horse_fake = g_z2h(x_zebra)

            # Generate recreational images
            x_zebra_rec = g_h2z(x_horse_fake)
            x_horse_rec = g_z2h(x_zebra_fake)

            if "buffer_size" in config and config["buffer_size"]:
                x_horse_fake_t = fake_h_buffer.push_and_pop(x_horse_fake)
                x_zebra_fake_t = fake_z_buffer.push_and_pop(x_zebra_fake)
            else:
                x_horse_fake_t = x_horse_fake
                x_zebra_fake_t = x_zebra_fake

            # Update discriminator 
            d_opt.zero_grad()
            d_loss = real_loss(d_h(x_horse))
            d_loss += fake_loss(d_h(x_horse_fake_t.detach()))
            d_loss += real_loss(d_z(x_zebra))
            d_loss += fake_loss(d_z(x_zebra_fake_t.detach()))
            d_loss.backward()
            d_opt.step()

            # Update generator
            g_opt.zero_grad()
            g_loss_fool = real_loss(d_h(x_horse_fake))
            g_loss_fool += real_loss(d_z(x_zebra_fake))
            g_loss_fool *= config['g_loss_weight'][0]
            g_loss_cycle = im_loss_1(x_horse, x_horse_rec)
            g_loss_cycle += im_loss_1(x_zebra, x_zebra_rec)
            g_loss_cycle *= config['g_loss_weight'][1]
            g_loss_iden = im_loss_2(g_h2z(x_zebra), x_zebra)
            g_loss_iden += im_loss_2(g_z2h(x_horse), x_horse)
            g_loss_iden *= config['g_loss_weight'][2]
            g_loss = g_loss_fool + g_loss_cycle + g_loss_iden
            g_loss.backward()
            g_opt.step()

            # Update batch losses
            logging['d_loss'] += d_loss.item()/len(data_zebra)
            logging['g_loss'] += g_loss.item()/len(data_zebra)
            logging['g_loss_fool'] += g_loss_fool.item()/len(data_zebra)
            logging['g_loss_cycle'] += g_loss_cycle.item()/len(data_zebra)
            logging['g_loss_iden'] += g_loss_iden.item()/len(data_zebra)

        # Make a visualization each epoch (logged to wandb)
        visualize_train(config, g_h2z, g_z2h, d_h, d_z, x_horse, x_zebra, plotting)
        
        # Save state every epoch
        save_state(g_h2z, g_z2h, d_h, d_z)

        # Log losses to wandb
        wandb.log(logging)
    
    # Finalize run
    wandb.finish()


def save_state(g_h2z, g_z2h, d_h, d_z):
    torch.save(g_h2z, path.join(save_folder, 'g_h2z.pt'))
    torch.save(g_z2h, path.join(save_folder, 'g_z2h.pt'))
    torch.save(d_h, path.join(save_folder, 'd_h.pt'))
    torch.save(d_z, path.join(save_folder, 'd_z.pt'))


def visualize_train(config, H2Z, Z2H, d_H, d_Z, x_horse, x_zebra, plotting=False):
    # Define image loss as L2 loss
    im_loss = gan_im_loss(config)
    lw = config['g_loss_weight']

    with torch.no_grad(): 
        # Generate fake images
        Z_fake = H2Z(x_horse)
        H_fake = Z2H(x_zebra)

        # Generate recreational images
        Z_rec = H2Z(H_fake)
        H_rec = Z2H(Z_fake)

        # Generate Identity images
        Z_iden = H2Z(x_zebra)
        H_iden = Z2H(x_horse)
        
        # Compute losses
        Z_fake_loss = lw[0]*fake_loss(d_Z(Z_fake)).cpu().numpy()
        H_fake_loss = lw[0]*fake_loss(d_H(H_fake)).cpu().numpy()
        Z_rec_loss = lw[1]*im_loss(x_zebra, Z_rec).cpu().numpy()
        H_rec_loss = lw[1]*im_loss(x_horse, H_rec).cpu().numpy()
        Z_iden_loss = lw[2]*im_loss(x_zebra, Z_iden).cpu().numpy()
        H_iden_loss = lw[2]*im_loss(x_horse, H_iden).cpu().numpy()

        # Convet to cpu device
        H_real = x_horse.cpu()
        Z_real = x_zebra.cpu()
        H_fake = H_fake.cpu()
        Z_fake = Z_fake.cpu()
        H_rec = H_rec.cpu()
        Z_rec = Z_rec.cpu()
        H_iden = H_iden.cpu()
        Z_iden = Z_iden.cpu()

    # Plot images
    n_rows = 1 if H_real.shape[0]<2 else 2 # How many rows should be shown

    # Show random images from the batch
    idx = np.random.randint(0,H_real.shape[0],(2,n_rows))

    f,ax = plt.subplots(n_rows*2, 4, figsize=(8, n_rows*5))
    for i in range(n_rows):
        # Horses
        ax[2*i,0].imshow(np.swapaxes(np.swapaxes((H_real[idx[0,i]].numpy()+1)/2,0,2),0,1))
        ax[2*i,0].axis('off')
        ax[2*i,0].set_title('Original')

        ax[2*i,1].imshow(np.swapaxes(np.swapaxes((Z_fake[idx[0,i]].numpy()+1)/2,0,2),0,1))
        ax[2*i,1].axis('off')
        ax[2*i,1].set_title('Fake, d={:.2f}'.format(Z_fake_loss))

        ax[2*i,2].imshow(np.swapaxes(np.swapaxes((H_rec[idx[0,i]].numpy()+1)/2,0,2),0,1))
        ax[2*i,2].axis('off')
        ax[2*i,2].set_title(f'Recovered, d={np.round(H_rec_loss,2)}')

        ax[2*i,3].imshow(np.swapaxes(np.swapaxes((H_iden[idx[0,i]].numpy()+1)/2,0,2),0,1))
        ax[2*i,3].axis('off')
        ax[2*i,3].set_title(f'Identity, d={np.round(H_iden_loss,2)}')

        # Zebras
        ax[2*i+1,0].imshow(np.swapaxes(np.swapaxes((Z_real[idx[1,i]].numpy()+1)/2,0,2),0,1))
        ax[2*i+1,0].axis('off')
        ax[2*i+1,0].set_title('Original')

        ax[2*i+1,1].imshow(np.swapaxes(np.swapaxes((H_fake[idx[1,i]].numpy()+1)/2,0,2),0,1))
        ax[2*i+1,1].axis('off')
        ax[2*i+1,1].set_title('Fake, d={:.2f}'.format(H_fake_loss))

        ax[2*i+1,2].imshow(np.swapaxes(np.swapaxes((Z_rec[idx[1,i]].numpy()+1)/2,0,2),0,1))
        ax[2*i+1,2].axis('off')
        ax[2*i+1,2].set_title(f'Recovered, d={np.round(Z_rec_loss,2)}')

        ax[2*i+1,3].imshow(np.swapaxes(np.swapaxes((Z_iden[idx[1,i]].numpy()+1)/2,0,2),0,1))
        ax[2*i+1,3].axis('off')
        ax[2*i+1,3].set_title(f'Identity, d={np.round(Z_iden_loss,2)}')

    f.savefig('log_image.png', transparent=True, bbox_inches='tight')
    if plotting == True:
        display.clear_output(wait=True)
        plt.show()
    else:
        plt.close()
    wandb.log({"Train Visualization": wandb.Image("log_image.png")})
    return None
