## Chamfer Distance

### Build
```
conda activate [name_of_conda_environment]
cd chamfer_distance
python setup.py install
```

### Usage
```
import dist_chamfer as chamfer
chamfer_distance = chamfer.chamferDist()

dist1, dist2 = chamfer_distance(xyz1, xyz2)
```