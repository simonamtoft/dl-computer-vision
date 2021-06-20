import os
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, ToPILImage
from torchvision.utils import save_image
from data import HORSES, ZEBRAS

batch_size = 64
save_dest = "generated_data"

if torch.cuda.is_available():
    print("The code will run on GPU.")
else:
    print("The code will run on CPU.")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not os.path.exists(save_dest):
    os.makedirs(os.path.join(save_dest, "horses", "generated"))
    os.makedirs(os.path.join(save_dest, "horses", "source"))
    os.makedirs(os.path.join(save_dest, "zebras", "generated"))
    os.makedirs(os.path.join(save_dest, "zebras", "source"))

d_h = torch.load("saved_states/d_h.pt", map_location=device)
d_z = torch.load("saved_states/d_z.pt", map_location=device)
g_h2z = torch.load("saved_states/g_h2z.pt", map_location=device)
g_z2h = torch.load("saved_states/g_z2h.pt", map_location=device)

z_test_loader = DataLoader(ZEBRAS(dataset="test", transform=ToTensor()), batch_size=batch_size, num_workers=4)
h_test_loader = DataLoader(HORSES(dataset="test", transform=ToTensor()), batch_size=batch_size, num_workers=4)

g_h2z.eval()
g_z2h.eval()
with torch.no_grad():
    ## Generate fake horses
    i = 0
    for Z in z_test_loader:
        orig = Z
        Z = Z.to(device)*2 - 1
        H = (g_z2h(Z) + 1) / 2
        for j, h in enumerate(H):
            save_image(h, os.path.join(save_dest, "horses", "generated", f"{i}.png"))
            save_image(orig[j], os.path.join(save_dest, "horses", "source", f"{i}.png"))
            i += 1

    ## Generate fake zebras
    i = 0
    for H in h_test_loader:
        orig = H
        H = H.to(device)*2 - 1
        Z = (g_h2z(H) + 1) / 2
        for j, z in enumerate(Z):
            save_image(z, os.path.join(save_dest, "zebras", "generated", f"{i}.png"))
            save_image(orig[j], os.path.join(save_dest, "zebras", "source", f"{i}.png"))
            i += 1
