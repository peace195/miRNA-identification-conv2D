## Precursor microRNA Identification

Code for project "Overview of Biomedicine" under lab course "Deep Learning for Computer Vision and Biomedicine" - Technical University of Munich, Germany. Depends on **Python 3, numpy, pytorch, scikit-learn, Bio, RNA, matplotlib**.

Our paper: Binh Thanh Do, Vladimir Golkov, Göktuğ Erce Gürel, and Daniel Cremers, "Precursor microRNA Identification Using Deep Convolutional Neural Networks"

* biorxiv: https://www.biorxiv.org/content/early/2018/09/16/414656

## Setting up the Environmental Variable

#### Ubuntu/Mac OSX
To set up the environmental variable in Mac OSX, follow the steps:
- Open `~/.bash_profile` with a text editor.
- Add the line `export VIENNA_PATH="the-path-of-your-viennarna-package"` and save.
- Restart the terminal, check if it is working by typing `echo $VIENNA_PATH`

An example: `export VIENNA_PATH="/Users/peace195/ViennaRNA/lib/python3.6/site-packages"`

## Usage
	
	python filename.py

where:

File / Folder | Description
-----|------------
dataset/ | `human`, `cross-species` and `new` RNA sequences
results/ 	| running log of each epoch in testset
weights/ | saved model parameters
cv.py| 5-fold cross validation for `human` and `cross-species` dataset using fixed-sized inputs ConvNet architecture
test.py | Train and test in `human` and `cross-species` dataset with fixed-sized inputs ConvNet architecture
test_new.py| Train and test in `new` dataset with fixed-sized inputs ConvNet architecture
cv_variable_size.py| 5-fold cross validation for `human` and `cross-species` dataset using variable-sized inputs ConvNet architecture
test_variable_size.py| Train and test in `human` and `cross-species` dataset with variable-sized inputs ConvNet architecture
test_new_variable_size.py| Train and test in `new` dataset with variable-sized inputs ConvNet architecture
utils.py| Read sequences, encode sequences and measurements
ConvNet.py| Some ConvNet architectures such as Alexnet, Resnet, etc.
statistics.py| Statistics of dataset


