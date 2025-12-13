import json
import os
import pandas as pd
import subprocess
from jinja2 import Template

"""
Classe responsável por gerar o relatório final em LaTeX.
Compila os resultados gerados na pasta output e gera texto dissertativo usando Jinja2.
"""
class Relatorio:

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.file_path_report = os.path.join(self.output_dir, "relatorio_final.tex")
        self.config = self._read_config()

    def _read_config(self) -> dict:
        path = os.path.join(self.output_dir, "config.json")
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        return {"freq": 7, "h": 12} # Default fallback

    def _read_csv(self, filename: str) -> pd.DataFrame:
        path = os.path.join(self.output_dir, filename)
        if os.path.exists(path):
            return pd.read_csv(path)
        return pd.DataFrame()

    def _get_q1_data(self):
        # Q1: ACF/PACF stats
        df = self._read_csv("q1_stats.csv")
        
        # Identificar picos sazonais com base na frequência configurada
        freq = self.config.get("freq", 7)
        seasonal_lags = [freq * i for i in range(1, 5)] # Ex: 7, 14, 21, 28
        
        peaks = []
        if not df.empty:
            for lag in seasonal_lags:
                if lag in df.index: 
                    # Check column Lag if index is not Lag
                    pass
                
                # Assuming 'Lag' column exists
                row = df[df['Lag'] == lag]
                if not row.empty:
                    lower = row['ACF_Lower_CI'].values[0]
                    upper = row['ACF_Upper_CI'].values[0]
                    if not (lower <= 0 <= upper):
                        peaks.append(lag)
        
        # Persistência
        persistence = 0
        if not df.empty:
            for i in range(1, len(df)):
                row = df.iloc[i]
                lower = row['ACF_Lower_CI']
                upper = row['ACF_Upper_CI']
                if not (lower <= 0 <= upper):
                    persistence += 1
                else:
                    break
        
        return {"peaks": peaks, "persistence": persistence, "freq": freq}

    def _get_q2_data(self):
        df = self._read_csv("q2_stationarity_results.csv")
        results = {}
        if not df.empty:
            # ADF
            adf_row = df[(df['Test'] == 'ADF') & (df['Metric'] == 'p-value')]
            if not adf_row.empty:
                results['adf_pvalue'] = adf_row['Value'].values[0]
            
            # KPSS
            kpss_row = df[(df['Test'] == 'KPSS') & (df['Metric'] == 'p-value')]
            if not kpss_row.empty:
                results['kpss_pvalue'] = kpss_row['Value'].values[0]
        return results

    def _get_q3_data(self):
        df = self._read_csv("q3_metrics.csv")
        if not df.empty:
            return df.iloc[0].to_dict()
        return {}

    def _get_q4_data(self):
        outliers_df = self._read_csv("q4_outliers.csv")
        metrics_df = self._read_csv("q4_metrics.csv")
        
        std_resid = 0
        if not metrics_df.empty:
            std_resid = metrics_df['std_resid'].values[0]
            
        return {
            "outliers_count": len(outliers_df),
            "outliers_list": outliers_df.to_dict('records'),
            "std_resid": std_resid
        }

    def _generate_latex_content(self) -> str:
        # Coletar dados
        q1_data = self._get_q1_data()
        q2_data = self._get_q2_data()
        q3_data = self._get_q3_data()
        q4_data = self._get_q4_data()

        # Template Jinja2
        template_str = r"""\documentclass[12pt, a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage{graphicx}
\usepackage{geometry}
\geometry{a4paper, margin=2.5cm}

\title{Relatório de Análise de Série Temporal: \\Nascimentos Femininos}
\author{Ubiratan da Silva Tavares}
\date{\today}

\begin{document}

\maketitle

\section{Introdução}
Este relatório apresenta a análise da série temporal de nascimentos femininos diários. O objetivo é compreender a dinâmica da série, identificar padrões como sazonalidade e tendência, diagnosticar a estacionariedade, estimar um modelo de previsão (Suavização Exponencial Simples - SES) e avaliar a presença de \textit{outliers}.

Para garantir a transparência e a replicabilidade dos resultados apresentados neste relatório, todo o código fonte encontra-se disponível publicamente no repositório do GitHub: \texttt{https://github.com/ubiratantavares/time-series-forecasting-SES-analysis}.

\section{Metodologia}
A análise foi conduzida utilizando a linguagem Python e um conjunto de bibliotecas especializadas para ciência de dados e estatística. A seguir, detalha-se a utilização de cada pacote no projeto:

\begin{itemize}
    \item \textbf{pandas}: Utilizado para a manipulação e estruturação dos dados em formato tabular (DataFrames). Fundamental para o tratamento da série temporal, permitindo indexação temporal, tratamento de dados faltantes e operações de fatiamento para divisão entre treino e teste.
    \item \textbf{numpy}: Empregado para operações numéricas de alto desempenho. Serve como base para cálculos matemáticos e vetoriais necessários durante a manipulação dos dados e cálculo de métricas.
    \item \textbf{statsmodels}: Biblioteca central para a modelagem econométrica e estatística. Foi utilizada para:
    \begin{itemize}
        \item Calcular as funções de autocorrelação (ACF) e autocorrelação parcial (PACF) na Questão 1.
        \item Realizar os testes de raiz unitária e estacionariedade (ADF e KPSS) na Questão 2.
        \item Estimar e ajustar o modelo de Suavização Exponencial Simples (SES) na Questão 3.
    \end{itemize}
    \item \textbf{matplotlib}: Utilizada para a geração de todas as visualizações gráficas do relatório, incluindo gráficos de linha da série temporal, correlogramas e gráficos de resíduos. Essencial para a inspeção visual dos resultados.
    \item \textbf{seaborn}: Utilizada para aprimorar a estética e o estilo das visualizações gráficas, garantindo gráficos mais informativos e visualmente agradáveis.
    \item \textbf{scikit-learn}: Empregada para o cálculo rigoroso das métricas de avaliação de acurácia do modelo. Forneceu as funções para cálculo do Erro Quadrático Médio (MSE), Erro Absoluto Médio (MAE) e Erro Percentual Absoluto Médio (MAPE).
    \item \textbf{jinja2}: Motor de templates utilizado para a geração automatizada deste relatório. Permite a inserção dinâmica dos resultados estatísticos, tabelas e textos interpretativos diretamente na estrutura do documento LaTeX.
\end{itemize}

A metodologia analítica seguiu as seguintes etapas:
\begin{enumerate}
    \item Análise de Autocorrelação (ACF/PACF) para identificação de sazonalidade e dependência temporal.
    \item Testes de Estacionariedade: \textit{Augmented Dickey-Fuller} (ADF) e KPSS.
    \item Ajuste de Modelo de Suavização Exponencial Simples (SES) e previsão fora da amostra.
    \item Diagnóstico de \textit{Outliers} utilizando o método de 3 Desvios Padrão (3-Sigma) nos resíduos.
\end{enumerate}

\section{Resultados e Discussões}

\subsection{Questão 1: Análise de Autocorrelação}

A Figura \ref{fig:q1_plot} apresenta os correlogramas da série.

\begin{figure}[htbp]
    \centering
    \includegraphics[width=1.0\textwidth]{q1_acf_pacf.png}
    \caption{Função de Autocorrelação (ACF) e Autocorrelação Parcial (PACF)}
    \label{fig:q1_plot}
\end{figure}

A análise dos correlogramas revela informações importantes sobre a estrutura da série temporal. 
{% if q1.peaks %}
Observam-se picos significativos nas defasagens sazonais ({{ q1.peaks }}), o que sugere fortemente a presença de um componente sazonal na série. Este padrão repetitivo indica que a série segue um ciclo regular (Frequência: {{ q1.freq }}).
{% else %}
Não foram observados picos significativos nas defasagens sazonais esperadas, sugerindo ausência de sazonalidade forte nesta frequência.
{% endif %}

Além disso, a análise da dependência temporal mostra que a autocorrelação permanece significativa para as primeiras {{ q1.persistence }} defasagens.
{% if q1.persistence > 5 %}
Isso indica uma alta persistência (memória longa), o que pode sugerir não-estacionariedade.
{% else %}
O decaimento rápido da função de autocorrelação sugere que a série possui uma dependência de curto prazo e tende à estacionariedade.
{% endif %}


\subsection{Questão 2: Testes de Estacionariedade}

Para confirmar as impressões visuais, foram realizados os testes formais ADF e KPSS.

\begin{table}[htbp]
    \centering
    \caption{Resultados dos Testes de Estacionariedade}
    \label{tab:q2_results}
    \begin{tabular}{lcc}
        \hline
        Teste & p-valor & Conclusão (a 5\%) \\
        \hline
        ADF & {{ "%.4f"|format(q2.adf_pvalue) }} & {% if q2.adf_pvalue < 0.05 %}Estacionária{% else %}Não Estacionária{% endif %} \\
        KPSS & {{ "%.4f"|format(q2.kpss_pvalue) }} & {% if q2.kpss_pvalue < 0.05 %}Não Estacionária{% else %}Estacionária{% endif %} \\
        \hline
    \end{tabular}
\end{table}

Os resultados indicam que:
{% if q2.adf_pvalue < 0.05 and q2.kpss_pvalue >= 0.05 %}
Ambos os testes concordam que a série é estacionária.
{% elif q2.adf_pvalue >= 0.05 and q2.kpss_pvalue < 0.05 %}
Ambos os testes concordam que a série NÃO é estacionária (possui raiz unitária ou tendência).
{% else %}
Há um conflito entre os testes. O teste ADF indica {% if q2.adf_pvalue < 0.05 %}estacionariedade{% else %}não estacionariedade{% endif %}, enquanto o teste KPSS sugere {% if q2.kpss_pvalue >= 0.05 %}estacionariedade{% else %}não estacionariedade{% endif %}. Isso pode indicar processos como estacionariedade por diferença ou tendência determinística.
{% endif %}

\subsection{Questão 3: Previsão com Suavização Exponencial Simples (SES)}

O modelo SES foi ajustado aos dados. A Figura \ref{fig:q3_plot} ilustra o ajuste e a previsão.

\begin{figure}[htbp]
    \centering
    \includegraphics[width=1.0\textwidth]{q3_forecast_plot.png}
    \caption{Previsão SES vs Dados Reais}
    \label{fig:q3_plot}
\end{figure}

O parâmetro de suavização ($\alpha$) estimado foi de {{ "%.4f"|format(q3.Alpha) }}. 
{% if q3.Alpha < 0.2 %}
Este valor baixo indica que o modelo considera um longo histórico passado, resultando em uma previsão suave e pouco reativa a flutuações recentes.
{% elif q3.Alpha > 0.8 %}
Este valor alto indica que o modelo reage fortemente às observações mais recentes (efeito memória curta).
{% else %}
Este valor intermediário indica um equilíbrio entre o histórico recente e passado.
{% endif %}

Quanto à acurácia, o modelo obteve um MAPE de {{ "%.2f"|format(q3.MAPE) }}\% e um RMSE de {{ "%.4f"|format(q3.RMSE) }}.
{% if q3.MAPE < 20 %}
O MAPE abaixo de 20\% sugere que o modelo possui uma boa capacidade preditiva para o horizonte testado.
{% else %}
O MAPE elevado sugere que o modelo SES pode não ser o mais adequado, possivelmente devido à presença de tendência ou sazonalidade não capturadas.
{% endif %}

O método SES, por projetar uma previsão constante, é teoricamente limitado para séries com tendência ou sazonalidade marcantes.

\subsection{Questão 4: Diagnóstico de Outliers}

A análise de resíduos (Figura \ref{fig:q4_plot}) permitiu identificar pontos atípicos.

\begin{figure}[htbp]
    \centering
    \includegraphics[width=1.0\textwidth]{q4_outliers_plot.png}
    \caption{Resíduos do Modelo SES e Outliers Detectados}
    \label{fig:q4_plot}
\end{figure}

Utilizando o critério de 3 desvios padrão, foram identificados {{ q4.outliers_count }} \textit{outliers}. 
A presença destes pontos, com um desvio padrão residual de {{ "%.4f"|format(q4.std_resid) }}, pode impactar a precisão das estimativas de erro (RMSE) e aumentar a incerteza das previsões. 
{% if q4.outliers_count > 0 %}
A natureza destes pontos deve ser investigada para determinar se são erros de coleta ou eventos reais atípicos.
{% else %}
A ausência de outliers estatísticos sugere que o modelo comporta-se de maneira estável em relação à variabilidade dos dados.
{% endif %}

\section{Conclusões}

Com base em todas as análises realizadas, conclui-se que o modelo SES apresenta um desempenho {% if q3.MAPE < 20 %}aceitável{% else %}limitado{% endif %} para previsões de curto prazo.
A análise exploratória indicou {% if q1.peaks %}presença{% else %}ausência{% endif %} de sazonalidade forte na frequência analisada.
Como o SES não modela explicitamente tendência nem sazonalidade, sua aplicação deve ser feita com cautela, especialmente para horizontes de previsão mais longos onde esses componentes estruturais dominariam.

\end{document}
"""
        
        template = Template(template_str)
        return template.render(q1=q1_data, q2=q2_data, q3=q3_data, q4=q4_data)

    def persist_results(self):
        latex_content = self._generate_latex_content()
        with open(self.file_path_report, 'w') as f:
            f.write(latex_content)
        print(f"Relatório LaTeX salvo em: {self.file_path_report}")
        return "Relatório final gerado com sucesso."

    def _compile_pdf(self):
        """
        Compila o arquivo .tex para .pdf usando pdflatex.
        """
        try:
            # Executa pdflatex duas vezes para garantir referências cruzadas corretas
            for _ in range(2):
                subprocess.run(
                    ["pdflatex", "-interaction=nonstopmode", "relatorio_final.tex"],
                    cwd=self.output_dir,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
            print(f"PDF gerado com sucesso em: {os.path.join(self.output_dir, 'relatorio_final.pdf')}")
        except subprocess.CalledProcessError as e:
            print("Erro ao compilar o PDF.")
            # Tentativa de decodificar com latin1, fallback para utf-8 ignorando erros
            try:
                print(e.stdout.decode('latin1'))
            except:
                print(e.stdout.decode('utf-8', errors='ignore'))

    def run(self):
        self.persist_results()
        self._compile_pdf()
