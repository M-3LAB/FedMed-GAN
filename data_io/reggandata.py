import glob
import torchvision.transforms as transforms
from data_io.base_class import BASE_DATASET, ToTensor

__all__ = ['RegganData']

class RegganData(BASE_DATASET):
    
    def __init__(self, root, modalities=["t1", "t2"], mode="train", extract_slice=[29, 100], transform_data=None, 
                    paired=True, clients=[1.0], seperated=True, regenerate_data=True, assigned_data=False, assigned_images=None):

        super(RegganData, self).__init__(root, modalities=modalities, mode=mode, extract_slice=extract_slice, 
                    paired=paired, clients=clients, seperated=seperated, regenerate_data=regenerate_data,
                    assigned_data=assigned_data, assigned_images=assigned_images)
        self.t = transform_data

        self._check_sanity()
        self._get_transform_modalities()
        self._generate_dataset()
        self._generate_client_indice()

    def _check_sanity(self):
        files_t1 = sorted(glob.glob("%s/%s/*" % (self.dataset_path, 'T1')))
        files_t2 = sorted(glob.glob("%s/%s/*" % (self.dataset_path, 'T2'))) 

        t1 = [f.split('/')[-1][:-4] for f in files_t1]
        t2 = [f.split('/')[-1][:-4] for f in files_t2]

        for x in t1:
            if x in t2:
                self.files.append(x)

    def _get_transform_modalities(self):
        if self.t:
            self.transform_a = transforms.Compose([transforms.ToPILImage(), 
                                                    transforms.RandomAffine(degrees=self.t[0]['degrees'], translate=self.t[0]['translate'], 
                                                                            scale=self.t[0]['scale'], fillcolor=self.t[0]['fillcolor']), 
                                                    transforms.Resize(size=self.t[0]['size']),
                                                    ToTensor()])

            self.transform_b = transforms.Compose([transforms.ToPILImage(), 
                                                    transforms.RandomAffine(degrees=self.t[1]['degrees'], translate=self.t[1]['translate'], 
                                                                            scale=self.t[1]['scale'], fillcolor=self.t[1]['fillcolor']), 
                                                    transforms.Resize(size=self.t[1]['size']), 
                                                    ToTensor()])
        else:
            self.transform_a = transforms.Compose([])
            self.transform_b = transforms.Compose([])




