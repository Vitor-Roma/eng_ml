import streamlit as st
import pandas
import numpy as np
from sklearn import model_selection, tree, ensemble, metrics, feature_selection
import joblib


fname = '../Data/dataset_kobe.csv'
savefile = '../Data/model_kobe.pkl'


############################################ LEITURA DOS DADOS
print('=> Leitura dos dados')
selected_features = {
    'minutes_remaining': int,
    'period': int,
    'playoffs': int,
    'shot_distance': int,
    'shot_made_flag': str,
    'shot_type': str,
}
df_kobe = pandas.read_csv('../Data/dataset_kobe.csv', sep=',',
                     usecols=list(selected_features.keys()),
                     converters = selected_features,
                     na_filter=True,
                     )
df_kobe = df_kobe.loc[ (df_kobe['shot_made_flag'] != "") ]
df_kobe['shot_made_flag'] = df_kobe['shot_made_flag'].astype(int)
kobe_target_col = 'shot_made_flag'
print(df_kobe.head())

############################################ TREINO/TESTE E VALIDACAO
results = {}
for kobe_type in df_kobe['shot_type'].unique():
    print('=> Training for kobe:', kobe_type)
    print('\tSeparacao treino/teste')
    kobe = df_kobe.loc[df_kobe['shot_type'] == kobe_type].copy()
    Y = kobe[kobe_target_col]
    X = kobe.drop([kobe_target_col, 'shot_type'], axis=1)
    ml_feature = list(X.columns)
    # train/test
    xtrain, xtest, ytrain, ytest = model_selection.train_test_split(X, Y, test_size=0.2)
    cvfold = model_selection.StratifiedKFold(n_splits=10, random_state=0, shuffle=True)
    print('\t\tTreino:', xtrain.shape[0])
    print('\t\tTeste :', xtest.shape[0])

    ############################################ GRID-SEARCH VALIDACAO CRUZADA
    print('\tTreinamento e hiperparametros')
    param_grid = {
        'max_depth': [3, 6],
        'criterion': ['entropy'],
        'min_samples_split': [2, 5],
        'n_estimators': [5, 10, 20],
        'max_features': ["auto", ],
    }
    selector = feature_selection.RFE(tree.DecisionTreeClassifier(),
                                     n_features_to_select=4)
    selector.fit(xtrain, ytrain)
    ml_feature = np.array(ml_feature)[selector.support_]

    model = model_selection.GridSearchCV(ensemble.RandomForestClassifier(),
                                         param_grid=param_grid,
                                         scoring='f1',
                                         refit=True,
                                         cv=cvfold,
                                         return_train_score=True
                                         )
    model.fit(xtrain[ml_feature], ytrain)

    print('\tAvaliação do modelo')
    threshold = 0.5
    xtrain.loc[:, 'probabilidade'] = model.predict_proba(xtrain[ml_feature])[:, 1]
    xtrain.loc[:, 'classificacao'] = (xtrain.loc[:, 'probabilidade'] > threshold).astype(int)
    xtrain.loc[:, 'categoria'] = 'treino'

    xtest.loc[:, 'probabilidade'] = model.predict_proba(xtest[ml_feature])[:, 1]
    xtest.loc[:, 'classificacao'] = (xtest.loc[:, 'probabilidade'] > threshold).astype(int)
    xtest.loc[:, 'categoria'] = 'teste'

    kobe = pandas.concat((xtrain, xtest))
    kobe[kobe_target_col] = pandas.concat((ytrain, ytest))
    kobe['target_label'] = ['Acertou' if t else 'Errou'
                            for t in kobe[kobe_target_col]]

    print('\t\tAcurácia treino:', metrics.accuracy_score(ytrain, xtrain['classificacao']))
    print('\t\tAcurácia teste :', metrics.accuracy_score(ytest, xtest['classificacao']))

    print('\tRetreinamento com histórico completo')
    model = model.best_estimator_
    model = model.fit(X[ml_feature], Y)

    results[kobe_type] = {
        'model': model,
        'data': kobe,
        'features': ml_feature,
        'target_col': kobe_target_col,
        'threshold': threshold
    }

joblib.dump(results, savefile, compress=9)
