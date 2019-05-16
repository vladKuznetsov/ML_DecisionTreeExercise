# ML_DecisionTree exercise detailign steps of the process
This is a simple project to demonstrate supervised machine learning using DesisionTree model for classification.  

1. Create a DecisionTree Model
2. Perform data clean-up (removing NoNs, converting categorical features to one-hot values, and creating ndarray) using Pandas and NumPy
3. Perform random sampling of cleaned data: 80% training, 20% testing. Train the model to evaluate performance. 
4. Apply PCA to this model to reduce dimensionality of the feature set.
5. Tune the model:
    - using cross-validation with 6-folds for different tree depths and display results on stdout
    - Apply GridSearch framework from Scikit-learn to this model: 
      Parameter tuning is performed with exhaustive grid search on tuples (tree-depth, PCA redaction to given
      number of parameters, random initial state) using accuracy, precision, and recall estimators
6. Results are presented on stdout and in a few graphs made with matplotlib

Requirements:  
   Python 3.7; compatibility with other versions is not tested. 
   Pandas, NumPy, Scikit-learn, Matplotlib.
   
Usage: 
  python main_MLDecisionTree.py  
  
Input data: 
  Change to full pathname of your file on line #135 in main_MLDecisionTree.py
 > fileName = "/Volumes/DATA_1TB/Safary_Downloads/ml_data.csv"
