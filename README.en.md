# FedMed-GAN
## Preliminary
> Dependency

<!-- pytorch 1.8.1\
autodiff-privacy 0.2, pip3 install autodiff-privacy\
xgboost, pip3 install xgboost\
matplotlib, pip3 install -U matplotlib\
opencv, pip3 install opencv-python\
tqdm, pip3 install tqdm\
yaml, pip3 install pyyaml\
nibbael, pip3 install nibabel\
SimpleITK, pip3 install SimpleITK\
mkl-fft, pip3 install mkl-fft\
kornia, pip3 install kornia, pip3 install kornia[x]
-->
```bash
conda install pytorch=1.8.1 torchvision torchaudio cudatoolkit=10.1 -c pytorch
```
```bash
pip3 install -r requirements.txt
```

> Generate dataset
```bash
python3 data_preprecess/brats2019.py
```
> Test data loader
```bash
python3 legacy_code/example_dataset_loader.py
```
> Prepare Statistics for FID
Calculate statistics for the custom dataset using command line tool. See [fid_stats.py](fid_stats.py) for implementation details.
```bash
python3 fid_stats.py --dataset 'ixi'  --source-domain 't2' --target-domain 'pd' --gpu-id 0
```

> BISE 
```bash
python3 bise_training.py --dataset 'ixi' --noise-type 'gaussian'
```
> Options
```bash
--fed-aggregate-method fed-psnr/fed-avg --gpu-id 1 --num-epoch 20 --num-round 10 
```
```bash
--noise-type 'severe' --reg-gan --auxiliary-rotation --auxiliary-translation --auxiliary-scaling --identity --diff-privacy
```
```bash
--debug --save-img --single-img-infer 
```
```bash
--save-model --load-model --load-model-dir './work_dir/centralized/ixi/Tue Jan 11 20:18:31 2022'
 ```

## Federated Training 
> BraTS2021 ['t1', 't2', 'flair']
```bash
python3 federated_training.py --dataset 'brats2021' --model 'cyclegan' --source-domain 't1' --target-domain 't2' --data-path '/disk1/medical/brats2021/training' --valid-path '/disk1/medical/brats2021/validation'
```

> IXI  ['t2', 'pd']
```bash
python3 federated_training.py --dataset 'ixi'  --model 'cyclegan' --source-domain 't2' --target-domain 'pd' --data-path '/disk1/medical/ixi' --valid-path '/disk1/medical/ixi'
```

## Centralized Training
> BraTS2021 ['t1', 't2', 'flair']
```bash
python3 centralized_training.py --dataset 'brats2021' --model 'cyclegan' --source-domain 't1' --target-domain 't2' --data-path '/disk1/medical/brats2021/training' --valid-path '/disk1/medical/brats2021/validation'
```

> IXI  ['t2', 'pd']
```bash
python3 centralized_training.py --dataset 'ixi' --model 'cyclegan' --source-domain 'pd' --target-domain 't2' --data-path '/disk1/medical/ixi' --valid-path '/disk1/medical/ixi'  
```