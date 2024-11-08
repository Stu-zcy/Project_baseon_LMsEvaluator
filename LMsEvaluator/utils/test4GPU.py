# This file is used to check if your device's GPU is available.
import torch

if __name__ == "__main__":
    # For M series of macOS.
    print(torch.backends.mps.is_available())
    print(torch.backends.mps.is_built())
    device = torch.device("mps")
    print(device)

    # For others.
    # print(torch.cuda.is_available())
    # device = torch.device("gpu")
    # print(device)
