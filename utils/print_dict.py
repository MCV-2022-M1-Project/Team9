"""
Usage:
  print_dict.py <dictName> 
  print_dict.py -h | --help
Options:
"""

from docopt import docopt
import pickle


if __name__ == "__main__":

    # read args
    args = docopt(__doc__)
    dict_name = args['<dictName>']


    with open(dict_name, 'rb') as fd:
        dict = pickle.load(fd)

    print (dict)
    print ('Dictionary contains {} elements'.format(len(dict)))
