import torch
import os
import random 

__all__ = ['IXITiny']

#TODO: Replace T1 with PD
class IXITiny(torch.utils.data.Dataset):
    """Dataset utility class.

    Args:
        root: (str) Path of the folder with all the images.
        mode : {'train' or 'test'} Part of the dataset that is loaded.
        paired: When paired is True, it means T1 and T2 is paired
    """
    def __init__(self, root, mode="train", paired=False):

        files = sorted(os.listdir(root))
        patient_id = list(set([i.split()[0] for i in files]))
        self.paired = paired 
        random.seed(3) 

        imgs = []
        
        if mode == "train":
            for i in patient_id[:int(0.8 * len(patient_id))]:
                if (
                    os.path.isfile(os.path.join(root, i + " - T1.pt")) and
                    os.path.isfile(os.path.join(root, i + " - T2.pt"))
                ):
                    imgs.append((os.path.join(root, i + " - T1.pt"),
                                 os.path.join(root, i + " - T2.pt")))

        elif mode == "test":
            for i in patient_id[int(0.8 * len(patient_id)):]:
                if (
                    os.path.isfile(os.path.join(root, i + " - T1.pt")) and
                    os.path.isfile(os.path.join(root, i + " - T2.pt"))
                ):
                    imgs.append((os.path.join(root, i + " - T1.pt"),
                                 os.path.join(root, i + " - T2.pt")))

        self.imgs = imgs

    def __getitem__(self, index):
        if self.paired:
            t1_path, t2_path = self.imgs[index]
        else:
            t1_path = self.imgs[index][0] 
            t2_path = self.imgs[random.randint(0, len(self.imgs) - 1)][1]

        t1 = torch.load(t1_path)[None, :, :]
        t2 = torch.load(t2_path)[None, :, :]
        return {"T1": t1, "T2": t2}

    def __len__(self):
        return len(self.imgs)