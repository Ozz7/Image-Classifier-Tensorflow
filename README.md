# **Image-Classifier**

Python command line application that can train an image classifier on a dataset , then predict new images using the trained model.

## Requirements

The Code is written in Python 3 . If you are using a lower version of Python you can upgrade using the pip package, ensuring you have the latest version of pip.

to upgrade Python

`pip install python --upgrade`

Additional Packages that are required are: Numpy, Pandas, Matplotlib, Pytorch, PIL and json.
You can download them using pip

`pip install numpy pandas matplotlib pil`

To intall Tensorflow head over to the Tensorflow site select your specs and follow the instructions given.

## Command Line Application

* Predict flower name from an image with predict.py along with the probability of that name. That is, you'll pass in a single image /path/to/image and return the flower name and class probability.
   
   - Basic usage: `python predict.py /path/to/image checkpoint`
   - Options
        - Return top K most likely classes: `python predict.py input checkpoint --top_k 3`
        - Use a mapping of categories to real names: `python predict.py input checkpoint --category_names cat_to_name.json`
        - Use GPU for inference: `python predict.py input checkpoint --gpu`

## JSON File

In order for the network to print out the name of the flower a .json file is required. By using a .json file the data can be sorted into folders with numbers and those numbers will correspond to specific names specified in the .json file.


