import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from abstract.analysis import Analysis

"""
Classe responsável por responder aos objetivos da Questão 4.
"""
class Questao4(Analysis):

    def __init__(self, serie: pd.Series, output_dir: str):
        self.serie = serie.dropna()
        self.output_dir = output_dir
        self.file_path_plot = os.path.join(self.output_dir, "q4_outliers_plot.png")
        self.file_path_interpretation = os.path.join(self.output_dir, "q4_interpretation.txt")
        self.file_path_outliers = os.path.join(self.output_dir, "q4_outliers.csv")

    def _fit_model(self):
        """
        Ajusta o modelo SES para obter os resíduos.
        Usamos o mesmo modelo da Questão 3 para consistência nas estimativas.
        """
        model = SimpleExpSmoothing(self.serie, initialization_method="estimated").fit()
        return model

    def _detect_outliers(self, residuals: pd.Series):
        """
        Detecta outliers usando o método de 3 desvios padrão (3-Sigma).
        """
        mean_resid = residuals.mean()
        std_resid = residuals.std()
        
        threshold_upper = mean_resid + 3 * std_resid
        threshold_lower = mean_resid - 3 * std_resid
        
        outliers = residuals[(residuals > threshold_upper) | (residuals < threshold_lower)]
        
        return outliers, threshold_upper, threshold_lower, mean_resid, std_resid

    def _plot_residuals(self, residuals: pd.Series, outliers: pd.Series, upper: float, lower: float):
        """
        Plota os resíduos e destaca os outliers.
        """
        plt.figure(figsize=(12, 6))
        plt.plot(residuals.index, residuals, label='Resíduos', color='blue', alpha=0.7)
        plt.scatter(outliers.index, outliers, color='red', label='Outliers', zorder=5)
        plt.axhline(y=upper, color='orange', linestyle='--', label='Limiar Superior (3σ)')
        plt.axhline(y=lower, color='orange', linestyle='--', label='Limiar Inferior (3σ)')
        plt.axhline(y=0, color='black', linewidth=0.5)
        plt.title('Diagnóstico de Outliers - Resíduos do Modelo SES')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.file_path_plot)
        plt.close()
        print(f"Gráfico de outliers salvo em: {self.file_path_plot}")

    def _interpret_results(self, outliers: pd.Series, std_resid: float) -> str:
        """
        Gera a interpretação dos resultados.
        """
        interpretation = "Questão 4: Interpretação do Diagnóstico de Outliers:\n\n"
        
        # 1. Método utilizado
        interpretation += "1. Método de Avaliação:\n"
        interpretation += "* Utilizou-se a análise dos resíduos do modelo SES (estimado na Q3).\n"
        interpretation += "* O critério para identificação de outliers foi o método de 3 Desvios Padrão (3-Sigma).\n"
        interpretation += "* Pontos onde os resíduos desviam mais de 3 vezes o desvio padrão da média foram considerados outliers.\n"
        interpretation += "\n"
        
        # 2. Pontos identificados
        interpretation += "2. Pontos Influentes / Outliers Identificados:\n"
        if outliers.empty:
            interpretation += "* Nenhum outlier foi identificado com o critério de 3-Sigma.\n"
        else:
            interpretation += f"* Foram identificados {len(outliers)} outliers:\n"
            for date, value in outliers.items():
                interpretation += f"     - Data: {date.strftime('%Y-%m-%d')}, Resíduo: {value:.4f}\n"
        interpretation += "\n"
        
        # 3. Natureza dos pontos (Erros vs Movimentos Reais)
        interpretation += "3. Natureza dos Pontos (Erros/Episódios Atípicos ou Movimentos Reais?):\n"
        if outliers.empty:
            interpretation += "* Como não há outliers estatísticos, a série parece comportar-se dentro da variabilidade esperada do modelo.\n"
        else:
            interpretation += "* A distinção entre erro e movimento real depende do contexto da série (ex: nascimentos).\n"
            interpretation += "* Picos isolados podem indicar eventos específicos (feriados, eventos naturais) ou erros de coleta.\n"
            interpretation += "* Se os outliers forem consecutivos, podem indicar uma mudança estrutural ou um evento duradouro.\n"
            interpretation += "* Dado que estamos analisando nascimentos, valores extremos podem ser reais (sazonalidade atípica) mas raros.\n"
        interpretation += "\n"

        # 4. Impacto na capacidade preditiva
        interpretation += "4. Impacto na Capacidade Preditiva:\n"
        interpretation += f"* O desvio padrão dos resíduos é {std_resid:.4f}.\n"
        if outliers.empty:
            interpretation += "* A ausência de outliers sugere que o modelo SES é robusto e estável para esta série.\n"
            interpretation += "* As estimativas de erro (RMSE/MAPE) são confiáveis e não estão sendo distorcidas por eventos extremos.\n"
        else:
            interpretation += "* A presença de outliers pode inflar as métricas de erro (especialmente RMSE, que penaliza erros grandes).\n"
            interpretation += "* Eles aumentam a incerteza das previsões (intervalos de confiança mais largos).\n"
            interpretation += "* Se os outliers forem eventos passados não recorrentes, eles podem não afetar a previsão futura pontual do SES (que pesa mais o recente),\n"
            interpretation += "mas se ocorreram recentemente, podem distorcer o nível estimado (smoothing level) e a previsão flat.\n"

        return interpretation

    def run(self):
        model = self._fit_model()
        residuals = model.resid
        outliers, upper, lower, mean, std = self._detect_outliers(residuals)
        
        self._plot_residuals(residuals, outliers, upper, lower)
        
        # Salvar lista de outliers
        if not outliers.empty:
            df_outliers = outliers.reset_index()
            df_outliers.columns = ['Date', 'Residual']
            df_outliers.to_csv(self.file_path_outliers, index=False)
            print(f"Lista de outliers salva em: {self.file_path_outliers}")
        else:
            print("Nenhum outlier encontrado para salvar em CSV.")
            # Criar arquivo vazio com cabeçalho para evitar erro no Relatorio
            pd.DataFrame(columns=['Date', 'Residual']).to_csv(self.file_path_outliers, index=False)

        # Salvar métricas (std_resid)
        file_path_metrics = os.path.join(self.output_dir, "q4_metrics.csv")
        pd.DataFrame([{"std_resid": std}]).to_csv(file_path_metrics, index=False)
        print(f"Métricas de outliers salvas em: {file_path_metrics}")

        # Salvar interpretação
        interpretation = self._interpret_results(outliers, std)
        with open(self.file_path_interpretation, 'w') as f:
            f.write(interpretation)
        print(f"Interpretação salva em: {self.file_path_interpretation}")
        



