# Team9

## Activation of the environment
In order to be able to run the scripts below, first it must be activated an environment containing the modules described in the requirements.txt file. To activate the environment, the following command has to be executed:

```
pip install virtualenv

python -m venv .   #inside this repo

source env/Scripts/activate

pip install -r requirements.txt


```

## Usage of the scripts 
### compute_descriptors.py: 
- This script creates a .pkl file containing a list of Image objects given database path. Image objects contain traits to describe each database image such as the image pixel information and the descriptor associated to it (histogram).

  Example of usage:

  Assuming that `inputDir` is the path of the folder containing the database, the following can be executed to compute the descriptors with the default values (generated pickle path: `./database.pkl`, histogramType: `GRAYSCALE` , nbins: `16`)

```
python compute_descriptors.py inputDir 
```

Another example could be: 
```
python compute_descriptors.py inputDir --nbins 16 --histogramType HSV 
```
The available option values can be displayed executing the help command:
```
python compute_descriptors.py --help
```

### compute_similarity.py: 
- This script computes the similarity between a set of query images and the database previously generated with ``compute_descriptors.py`` (database.pkl). In particular, it  obtains the K most similar images to each query as well as computing the MAPK of the entire set compared to the ground truth specified in the query folder. Moreover, if the option `--removeBG` is set to `HSV`, `OTSU` or `LAB`, it will remove the background of each query image and compute the similarity taking into account only the pixels classified as foreground of each query. Likewise, if this option is enabled (!=False), the Fscore measures of the background removal task masks will be outputted as well.

The output files is a .pkl file containing a list of lists with the K most similar images to each query. If the background removal option is enabled, the resulting binary masks will also be outputted in .png format in the directory specified by --saveResultsPath (although left as an option, it is encouraged to set a specific results path so that masks don't get created in the current directory)
  Example of usage:

  Assuming that `inputDir` is the path of the folder containing the queries, the following can be executed to compute the similarity with the default values
```
python compute_similarity.py inputDir 
```

Another example could be: 
```
python compute_similarity.py inputDir --distance X2 --K 10 --saveResultsPath ResultsDir  --GT False --removeBG OTSU
```
The flag GT must be set to False to test it without a ground truth. 
The available option values can be displayed executing the help command:
```
python compute_similarity.py --help
```
