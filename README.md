# ML_DecisionTree exercise detailign steps of the process
This is a simple project to demonstrate supervised machine learning using DesisionTree model for classification. 

1. Create a DecisionTree Model
2. Data clean-up  (removing NoNs, converting ll categorical features to one-hot values, and creating nparray)  using Pandas and NumPy
3. Random sampling of cleaned data: 80% training, 20% testing. The process is repeated using cross-validation with 6 "folds" 
4. Train model for different tree depths and display results on stdout
5. Apply PCA to this model to reduce dimentionality of the feature set
6. Train this model with cross-validation
7. Apply GridSearch mechanisn from Scikit-learn to this model: 
  Parameter tuning is performed with exaustive grid search on touples (tree-depth, PCA redaction to given number of parameters,random initial state) using accuracy, precision, and recall estimators
8. Results are presented on stdout and in a few graphs made with matplotlib

Requirements:  
   Python 3.7; compatibility with other versions is not tested. 
   Pandas, NumPy, Scikit-learn, Matplotlib.
   
Usage: 
  python main_MLDecisionTree.py  
  
Input data: 
  Change to full oathname ofyour file on line #135 in main_MLDecisionTree.py
 > fileName = "/Volumes/DATA_1TB/Safary_Downloads/ml_data.csv"
