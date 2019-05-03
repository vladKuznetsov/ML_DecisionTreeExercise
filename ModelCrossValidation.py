from sklearn.model_selection import cross_val_score
import sklearn

def computeCrossValidation (model, test_data, test_target):

    tt = sorted(sklearn.metrics.SCORERS.keys())

    scores = cross_val_score(model, test_data, test_target, cv=6, scoring='balanced_accuracy')
    print("e1132 Cross validation scores=",scores)
    print("e1606 Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    return