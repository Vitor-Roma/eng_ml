import os

import pandas as pd
import streamlit as st
import joblib
import pandas
import seaborn as sns
import matplotlib.pyplot as plt

os.environ['MLFLOW_TRACKING_URI'] = 'sqlite:///mlruns.db'  # Criando uma variável de ambiente
import mlflow

st.set_page_config(page_title='Kobe Model', page_icon="🏀", layout="centered", initial_sidebar_state="auto",
                   menu_items=None)

model_path = '../Data/model_kobe.pkl'

logged_model = 'runs:/e0bca20fddd14cb387ac81d3d188b4b5/sklearn-model'
loaded_model = mlflow.pyfunc.load_model(logged_model)

df_train = pd.read_parquet('../Data/operalization/base_train.parquet')
df_test = pd.read_parquet('../Data/operalization/base_test.parquet')
df_3pts = pd.read_parquet('../Data/operalization/base_3pts.parquet')

print(loaded_model)
# import ipdb; ipdb.set_trace()

st.sidebar.title('Painel de Controle')
st.sidebar.markdown(f"""
Controle do tipo do arremesso analisado e entrada de variáveis para avaliação de novos arremessos.
""")

st.sidebar.header('Tipo de arremesso analisado')
threePT_Field_Goal = st.sidebar.checkbox('3 Pontos')
shot_type = '2PT Field Goal' if threePT_Field_Goal else '3PT Field Goal'


@st.cache(allow_output_mutation=True)
def load_data(path):
    return joblib.load(path)

#
results = load_data(model_path)

model = results[shot_type]['model']
train_data = results[shot_type]['data']
features = results[shot_type]['features']
target_col = results[shot_type]['target_col']
idx_train = train_data.categoria == 'treino'
idx_test = train_data.categoria == 'teste'
train_threshold = results[shot_type]['threshold']

st.title(f"""
Sistema Online de Avaliação de Arremessos Tipo {'3 Pontos' if shot_type == '3PT Field Goal' else '2 Pontos'}
""")

st.markdown(f"""
Esta interface pode ser utilizada para a apresentação dos resultados
do modelo de classificação da qualidade de arremessos de 2 e 3 pontos,
segundo as variáveis utilizadas para caracterizar os arremessos.

O modelo selecionado ({shot_type}) foi treinado com uma base total de {idx_train.sum()} e avaliado
com {idx_test.sum()} novos dados (histórico completo de X arremessos.

Os arremessos são caracterizados pelas seguintes variáveis: {features}.
""")

st.sidebar.header('Entrada de Variáveis')
form = st.sidebar.form("input_form")
input_variables = {}

for cname in features:
    input_variables[cname] = (form.slider(cname.capitalize(),
                                          min_value=float(train_data[cname].astype(float).min()),
                                          max_value=float(train_data[cname].astype(float).max()),
                                          value=float(train_data[cname].astype(float).mean()))
                              )

form.form_submit_button("Avaliar")


@st.cache
def predict_user(input_variables):
    X = pandas.DataFrame.from_dict(input_variables, orient='index').T
    Yhat = model.predict_proba(X)[0, 1]
    return {
        'probabilidade': Yhat,
        'classificacao': int(Yhat >= train_threshold)
    }


user_arremesso = predict_user(input_variables)

if user_arremesso['classificacao'] == 0:
    st.sidebar.markdown("""Classificação:
    <span style="color:red">*Errou* a cesta</span>.
    """, unsafe_allow_html=True)
else:
    st.sidebar.markdown("""Classificação:
    <span style="color:green">*Acertou* a cesta</span>.
    """, unsafe_allow_html=True)


fignum = plt.figure(figsize=(6, 4))
for i in train_data.shot_made_flag.unique():
    sns.distplot(train_data[train_data[target_col] == i].probabilidade,
                 #                 label=train_data[train_data[target_col] == i].target_label,
                 label=train_data[train_data[target_col] == i].shot_made_flag,
                 ax=plt.gca())
# User wine
plt.plot(user_arremesso['probabilidade'], 2, '*k', markersize=3, label='Arremesso do Usuário')

plt.title('Resposta do Modelo para Arremessos Históricos')
plt.ylabel('Densidade Estimada')
plt.xlabel('Probabilidade de Acerto do Arremesso Alta')
plt.xlim((0, 1))
plt.grid(True)
plt.legend(loc='best')
st.pyplot(fignum)
