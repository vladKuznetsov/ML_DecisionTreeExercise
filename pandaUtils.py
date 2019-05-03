
import pandas as pd
import numpy  as np
from    typing  import List
from    pandas  import DataFrame

def readCSVFileToDataFrame(_fileName : str)->DataFrame:

    retVal = pd.read_csv(_fileName)
    return retVal

def convToNdarrays(_df:DataFrame, _columnsForDelete:List[str], _clLabelsCol: str)->(np.ndarray, np.ndarray):

    cols = _columnsForDelete
    cols.append(_clLabelsCol)
    dfOut = _df.drop(columns=cols)

    labels = _df[_clLabelsCol].values
    npOut  = dfOut.values

    return npOut, labels

def _cleanUpData(_df:DataFrame, _columnsWithTextNaNs : List[str])->DataFrame:

    values = {}
    for col in _columnsWithTextNaNs:
        values[col] = ""
    df0 = _df.fillna(value=values)
    df1 = df0.dropna(axis=0)

    return df1

'''
def _getValuesForOneHot(_df:DataFrame, _col : str)->List[str]:

    ll=_df[_col].drop_duplicates().tolist()

    retVal = []
    for elm in ll:
        if  len(elm) > 0:
            retVal.append(elm)

    return retVal
'''

def _convertToOneHot(_df:DataFrame, _col : str):

    df0 = DataFrame({_col: _df[_col].drop_duplicates().tolist()})

    print ("e1320 df0=", df0.values)
    tt = pd.get_dummies(df0,prefix=[_col], dummy_na=True)
    df1 = tt.join(df0)

    df2 = _df.merge(df1, how='outer', left_on=_col, right_on=_col)
    df3 = df2.drop(columns=[_col, _col+'_nan'])

    return df3

# this is the main function for reading data
def readAndPrepareData(fileName, _columnsToDelete :List[str], _columnsForOneHot: List[str], columnWithLabels:str, verbose=True)->(np.ndarray, np.ndarray):

    df = readCSVFileToDataFrame(fileName)

    if verbose:
        print("e1142 original features shape ", df.shape)
        print("e1143 original columns  ", df.columns)

    df = _cleanUpData(df, _columnsForOneHot)

    for col in _columnsForOneHot:
        df = _convertToOneHot(df, col)


    if  verbose:
        print("e1135 cleaned features shape  ", df.shape)
        print("e1136 cleaned columns ", df.columns)

    features, labels = convToNdarrays(df, _columnsToDelete, columnWithLabels)

    return features, labels

if  __name__ == "__main__":
    fileName = "/Volumes/DATA_1TB/Safary_Downloads/ml_data.csv"
    features, labels = readAndPrepareData(fileName, ['Unnamed: 0', 'ID'], ['mpr', 'nux'], 'outputs_class')

