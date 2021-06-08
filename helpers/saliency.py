import torch
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_saliency(img, model, img_shape=(128, 128)):
    model.eval()
    img.requires_grad_()

    # Forward pass through model
    output = model(img.to(device))

    # Catch the output
    output_idx = output.argmax()
    output_max = output[0, output_idx]

    # Do backpropagation to get the derivative of the output based on the image
    output_max.backward()

    # Retireve the saliency map and also pick the maximum value from channels on each pixel.
    # In this case, we look at dim=1. Recall the shape (batch_size, channel, width, height)
    saliency, _ = torch.max(img.grad.data.abs(), dim=1) 
    saliency = saliency.reshape(img_shape[0], img_shape[1])
    return saliency


def plt_saliency_img(img, model, N=50, img_shape=(128, 128), save_img=False):
    # convert image to proper shape
    img = img.reshape(1, 3, img_shape[0], img_shape[1])

    # compute avg saliency
    sal = 0
    for _ in range(N):
        sal += compute_saliency(img, model, img_shape)
    sal /= N

    # flatten image
    img = img.reshape(-1, img_shape[0], img_shape[1])

    # Visualize the image and the saliency map
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img.cpu().detach().numpy().transpose(1, 2, 0))
    ax[0].axis('off')
    ax[0].set_title('Original Image')
    ax[1].imshow(sal.cpu(), cmap='gray')
    ax[1].axis('off')
    ax[1].set_title('Saliency Map')
    plt.tight_layout()
    plt.show()
    plt.savefig()
