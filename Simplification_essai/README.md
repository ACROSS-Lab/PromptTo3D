# Mesh Simplification

The python script for "Surface Simplification Using Quadric Error Metrics, 1997" [[Paper]](http://www.cs.cmu.edu/~garland/Papers/quadrics.pdf)



## Environments
```
python>=3.10
scipy==1.11.3
numpy==1.26.0
scikit-learn==1.3.0
tqdm
```

## Usage

```
python simplification.py [-h] -i data/model1.obj [-v V] [-p P] [-optim] [-isotropic]
```
A simplified mesh will be output in `data/output/`.

### Parameters
- `-i`: Input file name [Required]
- `-v`: Target vertex number [Optional]
- `-p`: Rate of simplification [Optional (Ignored by -v) | Default: 0.5]
- `-optim`: Specify for valence aware simplification [Optional | Recommended]
- `-isotropic`: Specify for isotropic simplification [Optional]

