# FortuneCookieClassifier
Console-based (bottom-up) machine learning program to classify fortune cookie as either a prediction of one's future, or as simply a wise saying.

![picture alt](https://cdn1.tnwcdn.com/wp-content/blogs.dir/1/files/2016/08/fortune-cookie-796x398.jpg "Example of a message which would be classified with label 1")

## Brief Summary
Using a Naive Bayes model (with Laplace smoothing) in conjunction with MAP decision rule, picking class with highest probability. To get our features, we use a "bag of words" representation; given our finite vocabulary, each feature vector will contain a value of either 1 or 0 for particular word--indicates if word has occured in corresponding message or not, respectively. As previously mentioned, our classification for a given fortune cookie message depends on if we believe it is a wise saying (class 0) or a prediction of one's future (class 1).

In the code, we go through 4 general steps:
* Preprocess training and testing data into feature vectors.
* Train model on training features.
* Test model with testing features.
* Report results and time taken to test and train <br />

> Current training accuracy: _83%_ <br />
> Current testing accuracy: _50%_ <br />

- - - -

_Code originally developed for CptS 570 at Washington State University in Fall 2016._ <br /><br />
**Written by Luke Weber** <br />
**Created 11/04/2016**
