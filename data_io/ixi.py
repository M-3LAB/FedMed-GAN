import glob
from data_io.base_class import BASE_DATASET

__all__ = ['IXI']

class IXI(BASE_DATASET):
    def __init__(self, root, modalities=["t1", "t2"], mode="train", 
                 extract_slice=[29, 100], noise_type='normal', transform_data=None, 
                 clients=[1.0], data_mode='paired', 
                 splited=True, regenerate_data=True, assigned_data=False, 
                 assigned_images=None):

        super(IXI, self).__init__(root, modalities=modalities, mode=mode, extract_slice=extract_slice,
                                  noise_type=noise_type, data_mode=data_mode, transform_data=transform_data, 
                                  clients=clients, splited=splited, regenerate_data=regenerate_data)

        # infer assigned images
        self.assigned_data = assigned_data
        self.assigned_images = assigned_images 

        self._check_noise_type()        
        self._get_transform_modalities()

        if self.assigned_data:
            self.dataset = self.assigned_images
        else: 
            self._check_sanity()
            self._generate_dataset()
            self._generate_client_indice()

    def _check_noise_type(self):
        return super()._check_noise_type()

    def _get_transform_modalities(self):
        return super()._get_transform_modalities()

    def _check_sanity(self):
        files_t2 = sorted(glob.glob("%s/%s/*" % (self.dataset_path, 'T2')))
        files_pd = sorted(glob.glob("%s/%s/*" % (self.dataset_path, 'PD')))

        t2 = [f.split('/')[-1][:-4] for f in files_t2]
        pd = [f.split('/')[-1][:-4] for f in files_pd]

        for x in t2:
            if x in pd:
                self.files.append(x)

    def _generate_dataset(self):
        return super()._generate_dataset()

    def _generate_client_indice(self):
        return super()._generate_client_indice()