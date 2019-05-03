from sklearn import tree
from sklearn.pipeline import Pipeline
import FeatureReductionPCA


def buildModel(max_depth=None, random_state=1):
    model = tree.DecisionTreeClassifier(random_state=random_state, max_depth=max_depth)
    return model


def getPipeLine(components=None, max_depth=None, random_state=1):
    pca_filter = FeatureReductionPCA.getDimensinalityPCAReducer(components=components)
    clf = buildModel(max_depth=max_depth, random_state=random_state)
    pipel = Pipeline(steps=[('PCA', pca_filter), ('tree_model', clf)])
    return pipel

def check_param_grid(param_grid) -> bool:
    params = ['PCA__n_components',
              'tree_model__max_depth',
              'tree_model__random_state']
    retVal = True
    for param in params:
        tt = param_grid.get(param, None)
        if param_grid.get(param, None) is None:
            retVal = False
            print("e1103 param %s is missing in grid." % param)

    return retVal


pass

'''
>>> # You can set the parameters using the names issued
>>> # For instance, fit using a k of 10 in the SelectKBest
>>> # and a parameter 'C' of the svm
>>> anova_svm.set_params(anova__k=10, svc__C=.1).fit(X, y)
'''
