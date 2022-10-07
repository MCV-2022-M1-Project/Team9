# Team9
## Usage of the scripts 
### compute_descriptors.py: 
- This script creates a .pkl file containing a list of Image objects given database path. Image objects contain traits to describe each database image such as the image pixel information and the descriptor associated to it (histogram).

  Example of usage:

  Assuming that `inputDir` is the path of the folder containing the database, the following can be executed to compute the descriptors with the default values (generated pickle path: `./database.pkl`, histogramType: `GRAYSCALE` , nbins: `16`)

```
python3 compute_descriptors.py inputDir 
```
  An example specifying all the options explicitly would be:
```
python3 compute_descriptors.py inputDir --DBpicklePath ./database.pkl --histType GRAYSCALE --nbins 16
```
The available option values can be displayed executing the help command:
```
python3 compute_descriptors.py --help
```

### compute_similarity.py: 
- This script computes the similarity between a set of query images and the database previously generated with ``compute_descriptors.py``. In particular, it  obtains the K most similar images to each query as well as computing the MAPK of the entire set compared to the ground truth specified in the query folder. Moreover, if the option `--removeBG` is set to `True`, it will remove the background of each query image and compute the similarity taking into account only the pixels classified as foreground of each query. Likewise, if this option is enabled, the Fscore measures of the background removal task masks will be outputted as well.

The output files is a .pkl file containing a list of lists with the K most similar images to each query. If the background removal option is enabled, the resulting binary masks will also be outputted in .png format.

  Example of usage:

  Assuming that `inputDir` is the path of the folder containing the database and queryDir contains the query images and ground truth, the following can be executed to compute the similarity with the default values (distance: `L1`, K: `3` , picklePath: `./result.pkl`, DBpicklePath `database.pkl`, removeBG: `False`)

```
  python3 compute_similarity.py inputDir queryDir [--distance=<dist>] [--K=<k>] [--picklePath=<ppath>] [--DBpicklePath=<dbppath>] [--removeBG=<bg>]
```
The available option values can be displayed executing the help command:
```
python3 compute_similarity.py --help
```