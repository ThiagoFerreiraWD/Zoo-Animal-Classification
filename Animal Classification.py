#!/usr/bin/env python
# coding: utf-8

# ![animalsClassification.png](attachment:animalsClassification.png)

# # 1. Introdução
# 
# Dataset disponível na [UCI Machine Learning](https://archive.ics.uci.edu/ml/datasets/Zoo) utilizado para aplicação de algoritmos de classificação.
# 
# 
# 

# ### 1.1. Descrição do Dataset
# 
# O Dataset é composto por 101 registros, cada registro com a descrição de um animal, disponível em 16 colunas.
# Há 7 classes de animais: Mamíferos, Passáros, Répteis, Peixes, Anfíbios, Insetos e Invertebrados.

# ### 1.2. Descrição das Variáveis
# 
# 
# |       **Variável**      |                       **Descrição**                    |
# |-------------------------|:------------------------------------------------------:|
# |    **animal_name**      |      Nome do animal|
# |    **hair**             |      Variável Booleana                   |
# |    **feathers**         |      Variável Booleana         |
# |    **eggs**             |      Variável Booleana                                    |
# |    **milk**             |      Variável Booleana                        |
# |    **airbone**          |      Variável Booleana|
# |    **aquatic predator** |      Variável Booleana|
# |    **toothed	backbone**|      Variável Booleana                    |
# |    **breathes**         |      Variável Booleana                               |
# |    **venomous**         |      Variável Booleana                          |
# |    **legs**             |      Variável Booleana                                     |
# |    **tail**             |      Variável Booleana |
# |    **domestic**         |      Variável Booleana                        |
# |    **catsize**          |      Variável Booleana               |
# |    **class_type**       |      Variável Target, indica a classe a qual o animal pertence               |

# ### 1.3. Objetivo
#      
# 
# Verificar qual algoritmo obtém o melhor desempenho na previsão da classe dos animais, baseado nas características descritas nas variáveis acima.

# ### <center> ------------------------------------------------------------------------------------------------------------------ </center>

# # 2. Importação das Bibliotecas

# In[1]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings
import networkx as nx
import seaborn as sns
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')


# ### <center> ------------------------------------------------------------------------------------------------------------------ </center>

# # 3. Carregamento do Dataset

# In[2]:


df = pd.read_csv('zoo.csv')


# ### <center> ------------------------------------------------------------------------------------------------------------------ </center>

# # 4. Visualização Geral dos Dados

# In[3]:


df.head()


# In[4]:


# O DataFrame é composto por 101 registros (linhas) e 18 atributos (colunas).
df.shape


# In[5]:


# Informações básicas do DataFrame.
df.info()


# > #### Conclui-se que não há valores ausentes e todas as variáveis, exceto a "animal_name" são numéricas.

# In[6]:


# Verificação das variáveis numéricas.
df.describe()


# In[7]:


# Verificação da variável categórica
df.describe(include='O')


# > #### Nota-se a provável duplicação de um registro, iremos verificar mais adiante.

# In[8]:


# Criação e visualização da coluna "Classe" de acordo com o tipo de animal para posterior plotagem do Grafo.
df['Classe'] = 'Mammal'
df.loc[df['class_type'].values == 2, 'Classe'] = 'Bird'
df.loc[df['class_type'].values == 3, 'Classe'] = 'Reptile'
df.loc[df['class_type'].values == 4, 'Classe'] = 'Fish'
df.loc[df['class_type'].values == 5, 'Classe'] = 'Amphibian'
df.loc[df['class_type'].values == 6, 'Classe'] = 'Bug'
df.loc[df['class_type'].values == 7, 'Classe'] = 'Invertebrate'
df[['class_type', 'Classe']].head(3)


# In[9]:


# Plotagem do Grafo, indicando a espécie de cada animal contido na base de dados.
categoriaAnimais = [str(x) for x in sorted(df['Classe'].unique())]
categoriaAnimaisAresta = [('Animals', x) for x in categoriaAnimais]
especies = [df['animal_name'][df['Classe'] == x].tolist() for x in sorted(df['Classe'].unique())]
especies = [j for i in especies for j in i]
especiesAresta = []

for animal in especies:
    for categoria in sorted(df['Classe'].unique()):
        if animal in df['animal_name'][df['Classe'] == categoria].tolist():
            especiesAresta.append(tuple((str(categoria), animal)))

arestaEspecies = especiesAresta + categoriaAnimaisAresta

plt.figure(figsize=(15, 15))
graph = nx.Graph()
graph.add_edges_from(arestaEspecies, color='red')

options = {'node_size': 0.25, 'width': 0.2,}

position = nx.spring_layout(graph)
nx.draw(graph, with_labels=True, pos=position, font_size=13, **options)
plt.show()


# In[10]:


# Informações do grafo
print(nx.info(graph))


# <a name="a"></a>

# In[11]:


fig, ax = plt.subplots(figsize=(18, 4))

sns.countplot('Classe', data=df, palette='Greens_r', order = df['Classe'].value_counts().index)
plt.title('Quantidade de Animais por Classe', fontsize=25, pad=55)
ax.set_ylabel(''), ax.tick_params(axis='y', labelleft=False, left=None)

for axis in ['top', 'right', 'left']:    ax.spines[axis].set_color(None)
ax.spines['bottom'].set_linewidth(1.5)

for i in ax.patches:
    ax.annotate('{:,}'.format(int(i.get_height())),
    (i.get_x() + i.get_width() / 2, i.get_height()),
    ha='center', va='baseline', xytext=(0,1),
    textcoords='offset points', fontsize=25)
    
plt.show()


# In[12]:


# Correlação entre as variáveis
plt.figure(figsize=(18, 12))
sns.set(font_scale=1.25)
sns.heatmap(df.corr(), annot=True, cbar=False, linewidths=0.5, cmap='Greens_r')
plt.show()


# ### <center> ------------------------------------------------------------------------------------------------------------------ </center>

# # 5. Data Wrangling

# In[13]:


# Conforme analisado na Visualização Geral dos Dados, vimos que o registro "frog" aparenta estar duplicado, vamos checar.
df.loc[df['animal_name'] == 'frog']


# > #### Podemos observar que as características que descrevem o animal "frog" diferem na coluna "venomous", portanto, trata-se de animais diferentes e não excluíremos o registro da base.

# In[14]:


# Iremos deletar do DF as colunas "animal_classe" e "Classe", uma vez que não serão úteis para a aplicação dos algoritmos.
df.drop(['animal_name', 'Classe'], axis=1, inplace=True)


# In[15]:


# Nova visualização do DataFrame
df.head(2)


# ### <center> ------------------------------------------------------------------------------------------------------------------ </center>

# # 6. Preparação Machine Learning

# ### 6.1. Declaração das Variáveis Preditoras e Target.

# In[16]:


X = df.drop('class_type', axis=1)
y = df['class_type']


# ### 6.2. Escalonamento dos Valores

# In[17]:


scaler = MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)


# ### 6.3. Divisão das Variáveis entre Treino e Teste

# In[18]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# ### <center> ------------------------------------------------------------------------------------------------------------------ </center>

# # 7. Aplicação dos Modelos de Machine Learning

# ### 7.1. Logistic Regression

# In[19]:


from sklearn.linear_model import LogisticRegression
modeloRegressaoLogistica = LogisticRegression()
modeloRegressaoLogistica.fit(X_train, y_train);


# ### 7.2. Support Vector Machine

# In[20]:


from sklearn.svm import SVC
modeloSVM = SVC(C=3)
modeloSVM.fit(X_train, y_train);


# ### 7.3. Neural Network

# In[21]:


from sklearn.neural_network import MLPClassifier
modeloRedeNeural = MLPClassifier(hidden_layer_sizes=(64, 64))
modeloRedeNeural.fit(X_train, y_train);


# ### 7.4. KNN

# In[22]:


from sklearn import neighbors
modeloKNN = neighbors.KNeighborsClassifier()
modeloKNN.fit(X_train, y_train);


# ### 7.5. Random Forest

# In[23]:


from sklearn.ensemble import RandomForestClassifier

modeloRandomForest = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=0)
modeloRandomForest.fit(X_train, y_train);


# ### <center> ------------------------------------------------------------------------------------------------------------------ </center>

# # 8. Resultados

# In[24]:


# Verificação do score na base de teste.
accLog = modeloRegressaoLogistica.score(X_test, y_test) * 100
accSVM = modeloSVM.score(X_test, y_test) * 100
accRNN = modeloRedeNeural.score(X_test, y_test) * 100
accKNN = modeloKNN.score(X_test, y_test) * 100
accRFT = modeloRandomForest.score(X_test, y_test) * 100


# In[25]:


# Exibição dos Resultados.
print('-'*29)
print(f"{'Resultados':^29}")
print('-'*29)
print(f' Regressão Logística: {accLog:.2f}%')
print(f'                 SVM: {accSVM:.2f}%')
print(f'       Redes Neurais: {accRNN:.2f}%')
print(f'                 KNN: {accKNN:.2f}%')
print(f'       Random Forest: {accRFT:.2f}%')
print('-'*29)


# ### <center> ------------------------------------------------------------------------------------------------------------------ </center>

# # 9. Conclusões
# 
# 
# 
# * O escalonamento dos valores (em razão da coluna legs) surtiu efeito apenas nos algoritmos de Redes Neurais e KNN, não influenciando em nada nos demais;
# 
# 
# * O parâmetro C no algoritmo SVM, responsável por controlar a tolerância a erros foi ajustado para 3, uma vez que é o valor mínimo para se atingir o melhor resultado durante os testes nesta base de dados;
# 
# 
# * Foram criadas camadas ocultas no algoritmo de Redes Neurais, em razão da melhoria no desempenho do algoritmo;
# 
# 
# * Todos os algoritmos tiveram desempenho semelhante; e
# 
# 
# * A distribuição dos animais pode ser um problema ([imagem](#a)) , uma vez que há poucas espécies de anfíbios, répteis e insetos disponíves na base, principalmente se comparadas ao total de mamíferos.

# ### <center> ------------------------------------------------------------------------------------------------------------------ </center>

#  |                                                                                                                                               |                                                       Contatos                                                      |                                                                                                                         |
#  |:---------------------------------------------------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------:|
#  | <img width=40 align='center' alt='Thiago Ferreira' src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg" /> | <img width=40 align='center' src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg" /> | <img width=40 align='center' src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/facebook/facebook-original.svg" /> |
#  |                                            [Linkedin](https://www.linkedin.com/in/tferreirasilva/)                                            |                                    [Github](https://github.com/ThiagoFerreiraWD)                                    |                                [Facebook](https://www.facebook.com/thiago.ferreira.50746)                               |
#  |                                                                                                                                               |                                                Autor: Thiago Ferreira                                               |                                                                                                                         |
