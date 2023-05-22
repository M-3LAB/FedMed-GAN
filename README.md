# FedMed-GAN

## Preliminary
> Dependency

```bash
conda install pytorch=1.8.1 torchvision torchaudio cudatoolkit=10.1 -c pytorch
```
```bash
pip3 install -r requirements.txt
```
## Data Structure

    ├── BraTS2021
    │   ├── training
    │   │   ├── BraTS2021_00000
    │   │   ├── BraTS2021_00002
    │   │   └── ...
    │   ├── validation
    │   │   ├── BraTS2021_00001
    │   │   ├── BraTS2021_00013
    │   │   ├── ...
    │ 
    ...
    
    ├── IXI
    │   ├── PD
    │   │   ├── IXI002-Guys-0828-PD.nii.gz
    │   │   ├── IXI012-HH-1211-PD.nii.gz
    │   │   └── ...
    │   ├── T2
    │   │   ├── IXI002-Guys-0828-T2.nii.gz
    │   │   ├── IXI012-HH-1211-T2.nii.gz
    │   │   ├── ...
    │ 
    ...
> Generate dataset
```bash
python3 data_preprecess/brats2021.py
```

```bash
python3 data_preprecess/ixi.py
```

> Prepare Statistics for FID Calculate statistics. See [./fid_stats.py](fid_stats.py) for details.
```bash
python3 fid_stats.py --dataset 'ixi'  --source-domain 't2' --target-domain 'pd' --gpu-id 0
```

> Options. See [./configuration/config.py](configuration/config.py) for details.
```bash
--fed-aggregate-method 'fed-avg'/'fed-psnr' --num-epoch 20 --num-round 10 --gpu-id 1
```
```bash
--fid [default=true]
--noise-type 'slight'/'severe' [default='normal'] 
--identity [default=true]
--diff-privacy [default=true]
--reg-gan 
--auxiliary-rotation --auxiliary-translation --auxiliary-scaling
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
python3 federated_training.py --dataset 'brats2021' --model 'cyclegan' --source-domain 't1' --target-domain 'flair' --data-path '/disk1/medical/brats2021/training' --valid-path '/disk1/medical/brats2021/validation'
```

> IXI  ['t2', 'pd']
```bash
python3 federated_training.py --dataset 'ixi'  --model 'cyclegan' --source-domain 'pd' --target-domain 't2' --data-path '/disk1/medical/ixi' --valid-path '/disk1/medical/ixi'
```

## Centralized Training
> BraTS2021 ['t1', 't2', 'flair']
```bash
python3 centralized_training.py --dataset 'brats2021' --model 'cyclegan' --source-domain 't1' --target-domain 'flair' --data-path '/disk1/medical/brats2021/training' --valid-path '/disk1/medical/brats2021/validation'
```

> IXI  ['t2', 'pd']
```bash
python3 centralized_training.py --dataset 'ixi' --model 'cyclegan' --source-domain 'pd' --target-domain 't2' --data-path '/disk1/medical/ixi' --valid-path '/disk1/medical/ixi'  
```

