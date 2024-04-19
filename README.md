# oxheart-prototype

To install the required packages to compile the pipeline:

```bash
pip install -r requirements.txt
```

Vertex Ai service Agent

Questions to client:
* What is the expected distribution and range of values of oldpeak?

What would I do with more time?
* More comprehensive testing (come up with examples)
Better documentation (i.e. function docstrings)
* Maybe k-fold cross validation instead of train/test split and then retrain on full dataset before deploying since the dataset is tiny.
* Hyperparameter tuning/feature selection/model comparison if it was real world data where we didn't already get 100% acc with logistic regression.
* Consider deploying to endpoint rather than cloud run if it was going to be used more heavily (vertex ai endpoint doesn't scale to 0 though so for prototype cloud run is ideal as it scales to 0).