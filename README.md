# List of my codes in Machine Learning

Thanks to @Killy85 - For his README that i reuse plenty of parts here.
This repo is the synthesis of all the exercices asked during the Machine Learning course I followed at Ynov in 2019.

The lesson was given by Jeff Abrahamson @JeffAbrahamson, whom gave us the basics needed to understand properly the base mechanism of Machine Learning

To execute the examples, ensure you have installed Python3.6 and all the libraries listed in `requirements.txt`

For python version, you may run :

```console
$ python --version
```

which should return

```console
$ python --version
Python 3.6.7
```

or higher.

## 1/ Statistics pandas

During the first lesson, Jeff introduce us to [`python`](https://www.python.org/) and [`pandas`](https://pandas.pydata.org/).

The first one is a language really helpfull while doing Machine Learning. `pandas` is a library specialized in data treatment which may be of great help when sorting and analysing data.

This folder contain a python script and a csv data file. The script import data from the file and outputs a basic analysis of what we can see in it.

To run it, imagining you have a shell at the root of this git project, just type:

```console
$ cd 1
$ python test.py
```

test.py use a set of data wich is bit useless, so to get a correct example i made an other with bitcoin (cf bitcoin.py)

```console
$ cd 1
$ python bitcoin.py
```

## 2/ Linear Regression

[`linear regression`](https://en.wikipedia.org/wiki/Linear_regression) is a way for us to estimate a model as a linear function.

To do so we have to use the `gradient descent` algorithm which is aimed at finding a local minimal value.

Using this on our cost function, this will help to approximate the best values of θ0 and θ1 to minimize the cost function value.

So we did implement the gradient descent in 'gradient_descent.py' and use the one available in scikit learn in `gradient_descent_w_scikit.py`.

Those return the linear model we calculated before and a normalized respresentation of error there is between the points use to calculate and the actual model.

regression_lineaire.py is a test i made to understand what i needed to do
main.py is the correct python file to launch

You can see the work that has been made using the following command:

```console
$ cd 2
$ python main.py
```

## 3/ Logistic Regression

In order to explain classification problem, we then discover how to create `Logistic Regression` models helping us to classify elements accordingly to train set.

mnist.py is a test i made to understand what the point
classifieur.py is my first code i attempt to made. It gave me strange results so i restarted
logistik.py is my final code, using logistic regression librairy

I only made the OvO (One versus One) version because of the comprehension OvR problem i submit on the issues from the cours-ML' fork

## 4/ Infinitesimal calculs

We were introduced to infinitesimal calculus and had to do some calculation using python.

Two things to do:

- Approach the value **e** using the fact that if **a == e** then **ln(a) = 1**
  `calcul.py` is a simple calcul code that find a value at a precision rounded reducting the variance
  To launch it you just have to type the following in your shell:

```console
$ cd 4
$ python calcul.py
```

- Calculate the value of **e** locally using the Taylor's sequence

  `taylor.py` is a application of the taylor formula (with MacLaurin simplification) that give the log expression and the infinitesimal calcul
  To launch it you just have to type the following in your shell:

```console
$ cd 4
$ python taylor.py
```

## 5/ Recommendation

We then studied recommendation algorithms. Thoses algorithms aims a predicting content for users according to differents values.

There is 3 types of recommendations:

- Content-based recommendation
- Collaborative recommendation
- Knowledge-Based recommendation

We studied the first 2 of them, the third one being expensive and hard to apply.

The first script, `movielens.py` use an example of frequency words calcul with SVD algorithm and movielens librairy

We train a model, using [`surprise`](http://surpriselib.com/) which ,thanks to linear regression, is able to predict how a user will rate a film according to how he noted other ones, and the way other users scored them too. At first launch, you may have to download the corpus!

To launch it, type the following:

```console
$ cd 5
$ python movielens.py
```

The second one aim at creating a recommendation engine enabling us to choose **n** papers related to the one we choose. This is the system we may use if we manage a website and we wand to offer more papers to our users to read.

tf_idp.py show differents words and words frequency in a text

To run this, you just have to type:

```console
$ cd 5
$ python tf_idf.py
```

## 6/ Random Forest

It's a new ML Calcul allowing to get scatter of points in a area and differenciate it

Using a lib called [`graphviz`](https://www.graphviz.org/) we can visualize the tree of decision for each component of a load of data, try it (make sure you can write in the space where you are launchinh the code):

```console
$ cd 6
$ python random_forest.py
```

## 7/ SVM

Clustering system aiming to scatter points of an area and clust it with classifiers
Using [`SVC`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) from SVM lib

```console
$ cd 7
$ python svm.py
```

## 8/ Neuron Network

Simple test of tensorflow and keras.
I used a imdb datasets avalaible with keras library
It show the loss and the accuracy on imdb movie reviews

```console
$ cd 8
$ python tensorflou.py
```
