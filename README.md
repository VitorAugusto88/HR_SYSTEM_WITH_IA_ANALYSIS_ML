# TechNova HR — Sistema de Recrutamento com IA & Machine Learning

> Sistema inteligente de RH que combina **Machine Learning**, **NLP via Claude (Anthropic)** e **análise preditiva** para auxiliar no processo de recrutamento e seleção de candidatos.

---

## Visão Geral

O TechNova HR automatiza e enriquece o processo de entrevistas: gera perguntas personalizadas por vaga com IA, analisa as respostas do candidato com NLP e produz previsões sobre compatibilidade, fit cultural, risco de desligamento e perfil comportamental — tudo em uma interface web interativa construída com Streamlit.

---

## Funcionalidades

### Entrevista com IA
- Geração automática de **8 perguntas personalizadas** por vaga e cultura da empresa usando **Claude Haiku** (Anthropic)
- Distribuição equilibrada: 2 técnicas · 2 comportamentais · 2 de cultura · 2 situacionais
- **Follow-up dinâmico**: botão que aciona a IA para aprofundar uma resposta em tempo real
- Registro completo do candidato: nome, idade, anos de experiência, escolaridade e número de empresas anteriores

### Análise com Machine Learning
- **Previsão de attrition** (risco de desligamento) com XGBoost otimizado via Optuna, avaliado por predições out-of-fold
- **Clustering K-Means** para identificação automática do perfil comportamental do candidato
- **Compatibilidade com a vaga** calculada com pesos específicos por cargo (ex.: Eng. ML valoriza 50% técnico; PM valoriza 30% comunicação)
- **Fit cultural** com a TechNova Solutions
- **Potencial de crescimento** baseado nos scores extraídos da entrevista pelo **Claude Sonnet** (Anthropic)
- **Explicação dos fatores**: barras visuais mostrando o peso de cada competência no score final

### Comparação e Ranking
- Avalie múltiplos candidatos para a mesma vaga e compare lado a lado
- **Ranking automático** com Score Composto ponderado:
  ```
  Score = Compat. Vaga × 35% + Fit Cultural × 25% + Potencial × 25% + (100 − Risco Saída) × 15%
  ```
- Exportação do ranking em `.xlsx`

### Exportação de Relatório
- **Relatório individual em `.txt`** com métricas completas, pontos fortes, pontos de atenção e justificativa da recomendação (APROVADO / APROVADO COM RESSALVAS / REPROVADO)

### Dashboard Analytics
- 5 gráficos interativos sobre a base histórica de RH da TechNova
- Filtros por departamento e status de attrition
- Distribuição de idade, salário por departamento, taxa de attrition por área, dispersão Culture Fit × Engajamento e boxplots de scores comportamentais

---

## Vagas Disponíveis

| Cargo | Departamento | Nível | Faixa Salarial |
|---|---|---|---|
| Engenheiro(a) de Software | Technology | Pleno | R$ 8.000 – R$ 14.000 |
| Analista de Dados | Research & Development | Sênior | R$ 10.000 – R$ 18.000 |
| Gerente de Projetos | Sales | Sênior | R$ 12.000 – R$ 20.000 |
| Analista de RH | Human Resources | Pleno | R$ 5.000 – R$ 9.000 |
| Designer UX/UI | Research & Development | Pleno | R$ 7.000 – R$ 13.000 |
| Engenheiro(a) de Machine Learning | Research & Development | Sênior | R$ 15.000 – R$ 25.000 |
| Product Manager | Technology | Sênior | R$ 12.000 – R$ 22.000 |

---

## Modelos e Tecnologias de IA

| Componente | Tecnologia | Finalidade |
|---|---|---|
| Geração de perguntas e follow-ups | Claude Haiku (`claude-haiku-4-5-20251001`) | Rápido e eficiente para tarefas de geração |
| Análise de respostas e scoring NLP | Claude Sonnet (`claude-sonnet-4-6`) | Maior precisão na avaliação comportamental |
| Previsão de attrition | XGBoost + Optuna | Classificação de risco de desligamento |
| Perfil do candidato | K-Means | Segmentação em clusters comportamentais |
| Visualização dos clusters | PCA (2 componentes) | Redução de dimensionalidade para gráficos |

### Diferenciais do treinamento (notebook)
- **SMOTE dentro do pipeline** (`ImbPipeline`): balanceamento da classe minoritária dentro de cada fold — sem vazamento de dados entre treino e validação
- **Optuna** (20 trials): otimização bayesiana de 9 hiperparâmetros do XGBoost com avaliação out-of-fold honesta
- **Threshold calibrado pela curva Precision-Recall**: ponto de corte escolhido no treino, nunca no teste
- **K automático**: número de clusters definido pelo Silhouette Score, sem arbitrariedade manual
- **Nomes de clusters dinâmicos**: atribuídos com base nos scores médios reais de cada grupo

---

## Estrutura do Projeto

```
TECH_NOVA_IA_SYSTEM/
│
├── app.py                              # Interface Streamlit — ponto de entrada
│
├── notebooks/
│   ├── 02_analise_exploratoria.ipynb   # EDA completa + treinamento dos modelos
│   └── old/                            # Versões anteriores (referência)
│
├── src/
│   ├── config.py                       # Cultura da empresa, vagas e pesos por cargo
│   ├── generate_dataset.py             # Gerador do dataset de RH sintético
│   ├── nlp_engine.py                   # Motor NLP com Claude (Anthropic)
│   ├── ml_predictor.py                 # Predições, explicações e ranking de candidatos
│   └── __init__.py
│
├── models/                             # Artefatos treinados — gerados pelo notebook
│   ├── attrition_model.joblib          # Pipeline XGBoost (SMOTE + modelo)
│   ├── attrition_threshold.joblib      # Threshold calibrado pela curva PR
│   ├── kmeans_model.joblib             # Modelo K-Means treinado
│   ├── scaler_cluster.joblib           # StandardScaler para clustering
│   ├── scaler_classification.joblib    # StandardScaler para classificação (Logistic Regression)
│   ├── label_encoders.joblib           # Encoders por coluna categórica
│   ├── feature_cols.joblib             # Lista de features do modelo de attrition
│   ├── cluster_features.joblib         # Lista de features do clustering
│   ├── pca_model.joblib                # PCA (2 componentes)
│   └── cluster_names.joblib            # Nomes dos clusters gerados automaticamente
│
├── data/
│   └── hr_dataset.csv                  # Dataset gerado por generate_dataset.py
│
├── assets/                             # Gráficos gerados pelo notebook (.png)
│
├── requirements.txt                    # Dependências com versões fixadas
├── .env                                # Chave de API (não versionado)
└── .gitignore
```

---

## Pré-requisitos

- Python 3.10 ou superior
- Chave de API da Anthropic — obtenha em [console.anthropic.com](https://console.anthropic.com)

---

## Instalação e Execução

### 1. Clone o repositório e crie o ambiente virtual

```bash
git clone <url-do-repositorio>
cd TECH_NOVA_IA_SYSTEM

python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 2. Instale as dependências

```bash
pip install -r requirements.txt
```

### 3. Configure a chave de API

Crie um arquivo `.env` na raiz do projeto:

Para guardar sua chave da Anthoipc.
ANTHROPIC_API_KEY=sua-chave-aqui
```
Para usa-la insira na sidebar do projeto após executa-lo.

### 4. Execute o app

```bash
streamlit run app.py
```

O app abre automaticamente em `http://localhost:8501`.

---

## Como Usar o Sistema

1. Insira sua **Anthropic API Key** na sidebar 
2. Na página **Início**, escolha uma vaga e clique em *Iniciar Entrevista*
3. Preencha os dados do candidato e aguarde a IA gerar as perguntas
4. Registre as respostas — use o botão **Follow-up** para aprofundar respostas quando necessário
5. Ao finalizar, visualize o resultado: recomendação, radar de competências, gauges de ML e explicação dos fatores
6. Baixe o **relatório em `.txt`** ou acesse **Comparar Candidatos** para ver o ranking entre entrevistados
7. Explore o **Dashboard Analytics** para insights sobre a base histórica de RH

---

## Dependências Principais

| Biblioteca | Versão | Uso |
|---|---|---|
| streamlit | 1.56.0 | Interface web interativa |
| anthropic | 0.95.0 | NLP com Claude Haiku e Sonnet |
| xgboost | 3.2.0 | Modelo de classificação de attrition |
| scikit-learn | 1.8.0 | Pré-processamento, clustering e métricas |
| imbalanced-learn | 0.14.1 | SMOTE dentro do pipeline |
| optuna | 4.8.0 | Otimização bayesiana de hiperparâmetros |
| plotly | 6.7.0 | Gráficos interativos |
| pandas | 3.0.2 | Manipulação de dados |
| numpy | 2.4.4 | Operações numéricas |
| matplotlib / seaborn | 3.10.8 / 0.13.2 | Gráficos do notebook |
| joblib | 1.5.3 | Serialização dos artefatos treinados |
| shap | 0.51.0 | Disponível para explicabilidade futura |
| nbconvert | 7.17.1 | Execução do notebook via linha de comando |

---


```


## Sobre a Empresa Configurada

**TechNova Solutions** — empresa de tecnologia com ambiente híbrido (3 dias presencial · 2 remoto), liderança servidora e foco em inovação, colaboração e diversidade.
A cultura, os valores e o perfil ideal de profissional da TechNova são usados diretamente nos prompts enviados ao Claude para gerar perguntas contextualizadas e avaliar o alinhamento de cada candidato com a organização.