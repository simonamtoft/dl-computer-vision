import json
from os import path
import numpy as np
import torch
import matplotlib.pyplot as plt
from IPython import display
import wandb

from helpers import gan_im_loss, ImageBuffer, gan_loss_func

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
save_folder = 'saved_states'

# Define loss functions as LSGAN
def real_loss(x):
    return torch.mean((x - 1)**2)

def fake_loss(x):
    return torch.mean(x**2)

def lambda_lr(n_epochs, offset, delay):
    """
    Creates learning rate step function for LambdaLR scheduler.
    Stepping starts after "delay" epochs and will reduce LR to 0 when "n_epochs" has been reached
    Offset is used continuing training models.
    """
    return lambda epoch: 1 - max(0, epoch + offset - delay)/(n_epochs - delay)


def train_cycle_gan(config, g_h2z, g_z2h, d_h, d_z, z_dl, h_dl, p_name='tmp', plotting=False):
    """Training function for the Cycle GAN network
    Inputs
        config  :   A config dict that is sent to weight and biases, 
                    and used to set different training hyperparameters
                        'lr_g'          :   Learning rate for the 
                                            generator optimizer
                        'lr_d'          :   Learning rate for the 
                                            discriminator optimizer
                        'loss_func'     :   GAN-loss function used.
                                            can be either ['lsgan',[a,b,c]]
                                            or ['minimax',-1].
                        'epochs'        :   Number of epochs to train
                        'img_loss'      :   Specify how to compute the 
                                            image losses as list of 
                                            strings 'l1' or 'l2':
                                            [cycele, identity]
                        'g_loss_weight' :   Specify the weighting of
                                            the generator losses as a list 
                                            of ints: [fool, cycle, identity]
                        'buf_size'      :   Size of the image buffer during 
                                            training for the generated zebra
                                            and horse images.
                        'lr_decay'      :   A dict for the learning rate
                                            scheduler with keys 'offset', 
                                            'delay' and 'n_epochs'.
        g_h2z   :   A generator nn.Module that converts horses to zebras
        g_z2h   :   A generator nn.Module that converts zebras to horses
        d_h     :   A discriminator nn.Module that can discriminate horses
        d_z     :   A discriminator nn.Module that can discriminate zebras
        z_dl    :   A Dataloader of the zebra training data
        h_dl    :   A Dataloader of the horse training data
        p_name  :   A string, determining the name of the project on wandb
    """
    print(f"\nStarting training with config:")
    print(json.dumps(config, sort_keys=False, indent=4))

    # Define image losses
    im_loss_1, im_loss_2  = gan_im_loss(config)
    GAN_loss = gan_loss_func(config)
    glw = config['g_loss_weight']

    # Initialize wandb run
    wandb.init(project=p_name, config=config)

    # Optimizers for generators and discriminators
    g_param = list(g_h2z.parameters()) + list(g_z2h.parameters())
    d_param = list(d_h.parameters()) + list(d_z.parameters())
    g_opt = torch.optim.Adam(g_param, lr=config["lr_g"], betas=(0.5, 0.999))
    d_opt = torch.optim.Adam(d_param, lr=config["lr_d"], betas=(0.5, 0.999))

    # Set scheduler for learning rate
    if "lr_decay" in config:
        g_sched = torch.optim.lr_scheduler.LambdaLR(g_opt, lr_lambda=lambda_lr(**config["lr_decay"]))
        d_sched = torch.optim.lr_scheduler.LambdaLR(d_opt, lr_lambda=lambda_lr(**config["lr_decay"]))
        print("Notice: Using LR decay.")
    else:
        print("Notice: Not using LR decay.")

    # Define image buffer
    if "buf_size" in config and config["buf_size"]:
        h_buffer = ImageBuffer(config["buf_size"])
        z_buffer = ImageBuffer(config["buf_size"])
        print("Notice: Using image buffer.")
    else:
        print("Notice: Not using image buffer.")

    # perform training
    print('\nStart of training loop\n')
    for epoch in range(config['epochs']):
        print(f"Epoch {epoch+1}/{config['epochs']}")

        # Converte loaders to iterators
        data_z = iter(z_dl)
        data_h = iter(h_dl)

        # Define logging dict for wandb
        logging = {
            'd_loss': 0,
            'g_loss': 0,
            'g_loss_fool': 0,
            'g_loss_cycle': 0,
            'g_loss_iden': 0,
        }

        # Go over all batches
        n_z = len(data_z)
        for i in range(n_z):
            # Get zebras
            x_z = next(data_z).to(device) * 2 - 1
            
            # Get horses
            # Using hack to keep getting horse images, even though there are less of them
            # https://discuss.pytorch.org/t/infinite-dataloader/17903/10
            try: 
                x_h = next(data_h).to(device) * 2 - 1
            except StopIteration:
                data_h = iter(h_dl)
                x_h = next(data_h).to(device) * 2 - 1

            # Generate fake images
            x_z_fake = g_h2z(x_h)
            x_h_fake = g_z2h(x_z)

            # Generate recreational images
            x_z_rec = g_h2z(x_h_fake)
            x_h_rec = g_z2h(x_z_fake)

            if "buf_size" in config and config["buf_size"]:
                x_h_fake_t = h_buffer.push_and_pop(x_h_fake)
                x_z_fake_t = z_buffer.push_and_pop(x_z_fake)
            else:
                x_h_fake_t = x_h_fake
                x_z_fake_t = x_z_fake

            # Update discriminator 
            d_opt.zero_grad()
            d_l = GAN_loss.discriminator(d_h, x_h, x_h_fake_t)
            #d_l = real_loss(d_h(x_h))
            #d_l += fake_loss(d_h(x_h_fake_t.detach()))
            d_l += GAN_loss.discriminator(d_z, x_z, x_z_fake_t)
            #d_l += real_loss(d_z(x_z))
            #d_l += fake_loss(d_z(x_z_fake_t.detach()))
            d_l.backward()
            d_opt.step()

            # Update generator
            g_opt.zero_grad()
            g_l_fool = (GAN_loss.generator(d_h, 0, x_h_fake)+GAN_loss.generator(d_z, 0, x_z_fake))* glw[0]
            #g_l_fool = (real_loss(d_h(x_h_fake)) + real_loss(d_z(x_z_fake))) * glw[0]
            g_l_cycle = (im_loss_1(x_h, x_h_rec) + im_loss_1(x_z, x_z_rec)) * glw[1]
            g_l_iden = (im_loss_2(g_h2z(x_z), x_z) + im_loss_2(g_z2h(x_h), x_h)) * glw[2]
            g_l = g_l_fool + g_l_cycle + g_l_iden
            g_l.backward()
            g_opt.step()

            # Update batch losses
            logging['d_loss'] += d_l.item()/n_z
            logging['g_loss'] += g_l.item()/n_z
            logging['g_loss_fool'] += g_l_fool.item()/n_z
            logging['g_loss_cycle'] += g_l_cycle.item()/n_z
            logging['g_loss_iden'] += g_l_iden.item()/n_z

        # Step learning rate scheduler
        if 'lr_decay' in config:
            g_sched.step()
            d_sched.step()
            logging['lr_g'] = g_sched.get_lr()[0]
            logging['lr_d'] = d_sched.get_lr()[0]

        # Make a visualization each epoch (logged to wandb)
        visualize_train(im_loss_1, im_loss_2, GAN_loss, g_h2z, g_z2h, d_h, d_z, x_h, x_z, glw, plotting)
        
        # Save state every epoch
        save_state(g_h2z, g_z2h, d_h, d_z)

        # Log losses to wandb
        wandb.log(logging, commit=True)
    
    # upload the model to wandb
    wandb.save(path.join(save_folder, '*.pt'))
    
    # Finalize run
    wandb.finish()
    return None


def save_state(g_h2z, g_z2h, d_h, d_z):
    torch.save(g_h2z, path.join(save_folder, 'g_h2z.pt'))
    torch.save(g_z2h, path.join(save_folder, 'g_z2h.pt'))
    torch.save(d_h, path.join(save_folder, 'd_h.pt'))
    torch.save(d_z, path.join(save_folder, 'd_z.pt'))
    return None


def visualize_train(im_loss_1, im_loss_2, GAN_loss, g_h2z, g_z2h, d_h, d_z, x_h, x_z, glw, plotting=False):
    # Func to fix images before imshow
    def fix_img(x):
        return np.swapaxes(np.swapaxes((x.cpu().numpy() + 1)/2, 0, 2), 0, 1)

    # Select images from batches to show
    # The batch-shapes might be different
    idx = np.arange(0, min([x_h.shape[0], x_z.shape[0]]))
    np.random.shuffle(idx)
    idx = idx[:2] if len(idx)>1 else idx[:1]

    with torch.no_grad(): 
        # Generate fake images
        z_fake = g_h2z(x_h)
        h_fake = g_z2h(x_z)

        # Generate recreational images
        z_rec = g_h2z(h_fake)
        h_rec = g_z2h(z_fake)

        # Generate Identity images
        z_iden = g_h2z(x_z)
        h_iden = g_z2h(x_h)
        
        # Compute losses
        z_fake_loss = glw[0]*GAN_loss.generator(d_z, 0, z_fake).cpu().numpy()
        h_fake_loss = glw[0]*GAN_loss.generator(d_h, 0, h_fake).cpu().numpy()
        #z_fake_loss = glw[0]*fake_loss(d_z(z_fake)).cpu().numpy()
        #h_fake_loss = glw[0]*fake_loss(d_h(h_fake)).cpu().numpy()
        z_rec_loss = glw[1]*im_loss_1(x_z, z_rec).cpu().numpy()
        h_rec_loss = glw[1]*im_loss_1(x_h, h_rec).cpu().numpy()
        z_iden_loss = glw[2]*im_loss_2(x_z, z_iden).cpu().numpy()
        h_iden_loss = glw[2]*im_loss_2(x_h, h_iden).cpu().numpy()

    # Plot images
    n_rows = len(idx)
    f, ax = plt.subplots(n_rows*2, 4, figsize=(8, n_rows*5))
    for i in range(n_rows):
        # Horses
        ax[2*i,0].imshow(fix_img(x_h[idx[i]]))
        ax[2*i,0].axis('off')
        ax[2*i,0].set_title('Original')
        
        ax[2*i,1].imshow(fix_img(z_fake[idx[i]]))
        ax[2*i,1].axis('off')
        ax[2*i,1].set_title('Fake, d={:.2f}'.format(z_fake_loss))
    
        ax[2*i,2].imshow(fix_img(h_rec[idx[i]]))
        ax[2*i,2].axis('off')
        ax[2*i,2].set_title(f'Recovered, d={np.round(h_rec_loss, 2)}')
        
        ax[2*i,3].imshow(fix_img(h_iden[idx[i]]))
        ax[2*i,3].axis('off')
        ax[2*i,3].set_title(f'Identity, d={np.round(h_iden_loss, 2)}')

        # Zebras
        ax[2*i+1,0].imshow(fix_img(x_z[idx[i]]))
        ax[2*i+1,0].axis('off')
        ax[2*i+1,0].set_title('Original')

        ax[2*i+1,1].imshow(fix_img(h_fake[idx[i]]))
        ax[2*i+1,1].axis('off')
        ax[2*i+1,1].set_title('Fake, d={:.2f}'.format(h_fake_loss))

        ax[2*i+1,2].imshow(fix_img(z_rec[idx[i]]))
        ax[2*i+1,2].axis('off')
        ax[2*i+1,2].set_title(f'Recovered, d={np.round(z_rec_loss, 2)}')

        ax[2*i+1,3].imshow(fix_img(z_iden[idx[i]]))
        ax[2*i+1,3].axis('off')
        ax[2*i+1,3].set_title(f'Identity, d={np.round(z_iden_loss, 2)}')

    f.savefig('log_image.png', transparent=True, bbox_inches='tight')
    if plotting == True:
        display.clear_output(wait=True)
        plt.show()
    else:
        plt.close()
    wandb.log({"Train Visualization": wandb.Image("log_image.png")}, commit=False)
    return None
