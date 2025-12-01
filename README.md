# Previsão e Diagnóstico em Séries Temporais

Este projeto implementa uma análise completa de séries temporais, focando na série de **Nascimentos Femininos Diários**.

O objetivo é explorar a dinâmica da série, testar estacionariedade, realizar previsões utilizando **Suavização Exponencial Simples (SES)** e diagnosticar outliers, culminando na geração automática de um relatório técnico em LaTeX.

O projeto foi desenvolvido como parte da Segunda Lista Prática para Nota – Previsão e Diagnóstico em Séries Temporais. O enunciado completo e os detalhes dos requisitos podem ser consultados em `docs/Lista_Pratica_2.pdf`.

## 📋 Funcionalidades

O sistema realiza as seguintes etapas de análise de forma automatizada:

1. **Análise de Autocorrelação (Questão 1)**:
    * Gera gráficos de ACF (Autocorrelação) e PACF (Autocorrelação Parcial).
    * Interpreta automaticamente a presença de sazonalidade e persistência temporal.
2. **Testes de Estacionariedade (Questão 2)**:
    * Executa os testes **Augmented Dickey-Fuller (ADF)** e **KPSS**.
    * Avalia se a série é estacionária ou possui raiz unitária.
3. **Previsão com SES (Questão 3)**:
    * Ajusta um modelo de Suavização Exponencial Simples (`SimpleExpSmoothing`).
    * Realiza previsões fora da amostra (horizonte configurável).
    * Calcula métricas de acurácia: **RMSE**, **MAE** e **MAPE**.
    * Interpreta o parâmetro de suavização ($\alpha$).
4. **Diagnóstico de Outliers (Questão 4)**:
    * Identifica outliers nos resíduos do modelo utilizando o critério de **3 Desvios Padrão (3-Sigma)**.
    * Gera lista de pontos atípicos e gráficos de resíduos.
5. **Relatório Automatizado**:
    * Compila todos os resultados, gráficos e interpretações.
    * Gera um arquivo **LaTeX** (`relatorio_final.tex`) pronto para compilação, contendo textos dissertativos gerados dinamicamente com **Jinja2**.

## 🚀 Como Executar

### Pré-requisitos

Certifique-se de ter o Python instalado (versão 3.8 ou superior). Instale as dependências listadas no arquivo `requirements.txt`:

```bash
pip install -r requirements.txt
```

As principais dependências são:

* `pandas`, `numpy`: Manipulação de dados.
* `statsmodels`: Modelagem estatística e testes.
* `matplotlib`, `seaborn`: Visualização de dados.
* `scikit-learn`: Métricas de avaliação.
* `jinja2`: Geração de templates para o relatório.

### Configuração

Os parâmetros da análise podem ser ajustados diretamente no arquivo `main.py`:

* **`freq`**: Frequência da sazonalidade (ex: `7` para dados diários com ciclo semanal).
* **`h`**: Horizonte de previsão (número de passos à frente, ex: `7`).

Ao executar o projeto, um arquivo `config.json` é gerado automaticamente na pasta `output/` para garantir que o relatório utilize os parâmetros corretos na interpretação dos resultados.

### Customização de Dados

Para utilizar seus próprios dados:

1. Coloque seu arquivo CSV na pasta `dataset/` (ou em outro local acessível).
2. Edite o arquivo `main.py`:
    * Atualize a variável `file_path` para apontar para seu novo arquivo.
    * Certifique-se de que o CSV tenha uma coluna de datas (para ser usada como índice) e uma coluna de valores.
    * Ajuste `serie.index.freq` conforme a frequência dos seus dados (ex: `'D'` para diário, `'MS'` para mensal).

### Execução

Para rodar a análise completa, execute o script principal na raiz do projeto:

```bash
python main.py
```

### Resultados

Após a execução, verifique a pasta `output/`. Ela conterá:

* **Gráficos**: `q1_acf_pacf.png`, `q3_forecast_plot.png`, `q4_outliers_plot.png`.
* **Dados**: Arquivos CSV com métricas e estatísticas (`q1_stats.csv`, `q3_metrics.csv`, etc.).
* **Interpretações**: Arquivos de texto com as conclusões parciais.
* **Relatório Final**: `relatorio_final.tex`.
  * Você pode compilar este arquivo usando qualquer editor LaTeX (Overleaf, TeXShop, etc.) ou via linha de comando (`pdflatex output/relatorio_final.tex`) para gerar o PDF final.

## 📂 Estrutura do Projeto

```text
.
├── controller/         # Lógica de controle e orquestração
│   └── controller.py
├── model/              # Implementação das análises (Questões 1-5 e Relatório)
│   ├── questao1.py     # Autocorrelação
│   ├── questao2.py     # Estacionariedade
│   ├── questao3.py     # Previsão SES
│   ├── questao4.py     # Outliers
│   ├── questao5.py     # Conclusão Geral
│   └── relatorio.py    # Geração do LaTeX com Jinja2
├── dataset/            # Dados de entrada
│   ├── daily-total-female-births.csv
│   └── daily-total-female-births.names.txt
├── output/             # Diretório onde os resultados são salvos
├── docs/               # Documentação e enunciados
│   └── Lista_Pratica_2.pdf
├── abstract/           # Classes abstratas
│   └── analysis.py     # Interface base para as análises
├── main.py             # Ponto de entrada da aplicação
├── requirements.txt    # Dependências do projeto
└── README.md           # Documentação
```

## 🛠️ Tecnologias Utilizadas

* **Linguagem**: Python 3
* **Análise de Séries Temporais**: Statsmodels
* **Templating**: Jinja2 (para geração de relatórios)
* **Arquitetura**: MVC (Model-View-Controller) simplificado

## ✒️ Autor

* **Ubiratan da Silva Tavares** - *Desenvolvimento e Análise*

## 📄 Licença

Este projeto é de uso educacional, desenvolvido para a disciplina de Modelagem Estatística. Sinta-se à vontade para estudar e modificar o código.
