import os
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
from data import HORSES, ZEBRAS
from pytorch_fid.fid_score import calculate_fid_given_paths

batch_size = 64
save_dest = "generated_data"
load_path = "saved_states"

if torch.cuda.is_available():
    print("The code will run on GPU.")
else:
    print("The code will run on CPU.")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not os.path.exists(save_dest):
    os.makedirs(os.path.join(save_dest, "horses", "train", "generated"))
    os.makedirs(os.path.join(save_dest, "horses", "train", "source"))
    os.makedirs(os.path.join(save_dest, "zebras", "train", "generated"))
    os.makedirs(os.path.join(save_dest, "zebras", "train", "source"))

    os.makedirs(os.path.join(save_dest, "horses", "test", "generated"))
    os.makedirs(os.path.join(save_dest, "horses", "test", "source"))
    os.makedirs(os.path.join(save_dest, "zebras", "test", "generated"))
    os.makedirs(os.path.join(save_dest, "zebras", "test", "source"))

d_h = torch.load(os.path.join(load_path, "d_h.pt"), map_location=device)
d_z = torch.load(os.path.join(load_path, "d_z.pt"), map_location=device)
g_h2z = torch.load(os.path.join(load_path, "g_h2z.pt"), map_location=device)
g_z2h = torch.load(os.path.join(load_path, "g_z2h.pt"), map_location=device)

z_test_loader = DataLoader(ZEBRAS(dataset="test", transform=ToTensor()), batch_size=batch_size, num_workers=8)
h_test_loader = DataLoader(HORSES(dataset="test", transform=ToTensor()), batch_size=batch_size, num_workers=8)
z_train_loader = DataLoader(ZEBRAS(dataset="train", transform=ToTensor()), batch_size=batch_size, num_workers=8)
h_train_loader = DataLoader(HORSES(dataset="train", transform=ToTensor()), batch_size=batch_size, num_workers=8)

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
            save_image(h, os.path.join(save_dest, "horses", "test", "generated", f"{i}.png"))
            save_image(orig[j], os.path.join(save_dest, "horses", "test", "source", f"{i}.png"))
            i += 1

    ## Generate fake zebras
    i = 0
    for H in h_test_loader:
        orig = H
        H = H.to(device)*2 - 1
        Z = (g_h2z(H) + 1) / 2
        for j, z in enumerate(Z):
            save_image(z, os.path.join(save_dest, "zebras", "test", "generated", f"{i}.png"))
            save_image(orig[j], os.path.join(save_dest, "zebras", "test", "source", f"{i}.png"))
            i += 1
    ## Generate fake horses
    i = 0
    for Z in z_train_loader:
        orig = Z
        Z = Z.to(device)*2 - 1
        H = (g_z2h(Z) + 1) / 2
        for j, h in enumerate(H):
            save_image(h, os.path.join(save_dest, "horses", "train", "generated", f"{i}.png"))
            save_image(orig[j], os.path.join(save_dest, "horses", "train", "source", f"{i}.png"))
            i += 1

    ## Generate fake zebras
    i = 0
    for H in h_train_loader:
        orig = H
        H = H.to(device)*2 - 1
        Z = (g_h2z(H) + 1) / 2
        for j, z in enumerate(Z):
            save_image(z, os.path.join(save_dest, "zebras", "train", "generated", f"{i}.png"))
            save_image(orig[j], os.path.join(save_dest, "zebras", "train", "source", f"{i}.png"))
            i += 1

fid_value = calculate_fid_given_paths([os.path.join(save_dest, "horses", "train", "generated"), os.path.join(save_dest, "zebras", "train", "source")], 50, device, 2048)
print('Training datasets')
print('Generated horses vs. real horses')
print('FID: ', fid_value)

fid_value = calculate_fid_given_paths([os.path.join(save_dest, "zebras", "train", "generated"), os.path.join(save_dest, "horses", "train", "source")], 50, device, 2048)
print('Training datasets')
print('Generated zebras vs. real zebras')
print('FID: ', fid_value)

fid_value = calculate_fid_given_paths([os.path.join(save_dest, "horses", "test", "generated"), os.path.join(save_dest, "zebras", "test", "source")], 50, device, 2048)
print('Test datasets')
print('Generated horses vs. real horses')
print('FID: ', fid_value)

fid_value = calculate_fid_given_paths([os.path.join(save_dest, "zebras", "test", "generated"), os.path.join(save_dest, "horses", "test", "source")], 50, device, 2048)
print('Test datasets')
print('Generated zebras vs. real zebras')
print('FID: ', fid_value)