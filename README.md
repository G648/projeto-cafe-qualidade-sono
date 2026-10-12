# Café e Qualidade do Sono — Projeto de Ciência de Dados

Projeto do curso de Ciência de Dados da **Alura**. A empresa fictícia **Health&Life Analytics** quer entender como o consumo de café influencia a **qualidade do sono** dos clientes e prever essa qualidade a partir dos hábitos de cada pessoa.

## Estrutura do repositório

```
├── projeto_cafe_sono.ipynb              # Notebook com as 3 entregas
├── data/
│   ├── synthetic_coffee_health_10000(in).csv   # Base original (10.000 clientes, 16 colunas)
│   └── coffee_health_processado.csv            # Base tratada e codificada para o modelo
├── modelo/
│   └── modelo_qualidade_sono.pkl               # Melhor modelo treinado (Regressão Logística)
├── requirements.txt
└── README.md
```

## Entregas

| # | Entrega | O que foi feito |
|---|---|---|
| 1 | **Análise exploratória** | Dimensões, tipos, nulos, duplicados, estatísticas, histogramas, boxplots, barras, pizzas e correlações |
| 2 | **Visualização e insights** | Gráficos comparativos, análises por gênero, idade e estresse, insights de negócio e seção *Principais Descobertas* |
| 3 | **Modelo preditivo** | Pré-processamento, feature derivada, Regressão Logística × Random Forest, avaliação, modelo salvo e *Recomendações para o Negócio* |

## Principais descobertas

- Clientes com **mais de 400 mg de cafeína/dia** dormem em média **27 minutos a menos** (6,24 h × 6,69 h).
- O sono ruim (*Poor*) é **2,3× mais comum** entre quem toma mais de 3 xícaras/dia (13% × 5,6%).
- **Estresse alto** = **2,8 h a menos de sono** que estresse baixo.
- Clientes estressados tomam **26% mais café**, em todas as faixas etárias.
- **Gênero e idade** quase não alteram o sono (~6,6 h em todos os grupos).

## Modelo

| Modelo | Acurácia treino | Acurácia teste |
|---|---|---|
| **Regressão Logística** | 97,5% | **97,0%** |
| Random Forest | 100% | 96,8% (leve overfitting) |

- A variável mais importante é `Sleep_Hours` (~80% da importância). Sem ela, o modelo cai para 56%, o mesmo que chutar sempre a classe mais comum.
- `Stress_Level` e `Health_Issues` foram removidas porque são derivadas das horas de sono (vazamento de dados).

### Usar o modelo salvo

```python
import joblib
import pandas as pd

modelo = joblib.load("modelo/modelo_qualidade_sono.pkl")
dados = pd.read_csv("data/coffee_health_processado.csv")

X = dados.drop(columns="Sleep_Quality")
print(modelo.predict(X.head()))
```

## Como executar

**Localmente (Jupyter):**
```bash
pip install -r requirements.txt
jupyter notebook projeto_cafe_sono.ipynb
```

**Google Colab:** faça upload do notebook e da pasta `data/` (mantendo o nome da pasta) e execute todas as células.

> O modelo `.pkl` foi salvo com **scikit-learn 1.9.1**. Para carregá-lo, use a mesma versão (já indicada no `requirements.txt`).

## Tecnologias

Python · pandas · Matplotlib · Seaborn · scikit-learn · joblib

## Dados

Base sintética [Global Coffee Health Dataset](https://www.kaggle.com/datasets/uom190346a/global-coffee-health-dataset) (Kaggle), fornecida pela Alura.
