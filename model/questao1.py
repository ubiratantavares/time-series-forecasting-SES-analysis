import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from abstract.analysis import Analysis

"""
Classe responsável por responder aos objetivos da Questão 1.
"""
class Questao1(Analysis):

    def __init__(self, serie: pd.Series, freq: int, output_dir: str):
        self.serie = serie.dropna()
        self.freq = freq
        self.output_dir = output_dir
        self.alpha = 0.05  # Define o nível de significância (95% CI)
        # Define lags como o menor múltiplo da frequência que é >= 40
        # Isso garante que os gráficos de ACF/PACF mostrem ciclos sazonais completos
        min_lags = 40
        self.lags = ((min_lags + self.freq - 1) // self.freq) * self.freq
        self.file_path_stats = os.path.join(self.output_dir, "q1_stats.csv")
        self.file_path_acf_pacf = os.path.join(self.output_dir, "q1_acf_pacf.png")

    # calcula a autocorrelação (ACF e PACF)
    def _calculate_autocorrelation(self) -> dict:
        # calcula ACF (Autocorrelação) e os CIs
        # Retorna: acf, confint, qstat, pvalues
        acf_values, acf_ci, qstat, pvalues = acf(self.serie, nlags=self.lags, alpha=self.alpha, fft=True, qstat=True)

        # calcula PACF (Autocorrelação Parcial) e os CIs
        pacf_values, pacf_ci = pacf(self.serie, nlags=self.lags, method='yw', alpha=self.alpha)

        return {
            "acf_values": acf_values,
            "pacf_values": pacf_values,
            "acf_ci": acf_ci,
            "pacf_ci": pacf_ci,
            "qstat": qstat,
            "pvalues": pvalues
        }

    # gera os correlogramas da autocorrelação (ACF e PACF)
    def _plot_acf_pacf(self):
        fig, axes = plt.subplots(1, 2, figsize=(15, 4))

        # plota ACF
        plot_acf(self.serie, lags=self.lags, ax=axes[0], title=f'Função de Autocorrelação (ACF) - Freq: {self.freq}')

        # plota PACF
        plot_pacf(self.serie, lags=self.lags, ax=axes[1], title='Função de Autocorrelação Parcial (PACF)')

        plt.tight_layout()
        plt.savefig(self.file_path_acf_pacf)
        plt.close(fig)
        print(f"Gráfico ACF/PACF salvo em: {self.file_path_acf_pacf}")

    def _save_stats(self, results: dict):
        # Create a DataFrame to save the statistics
        lags_index = range(len(results["acf_values"])) # 0 to nlags
        
        # Pad qstat and pvalues with NaN for lag 0 to match size
        qstat_padded = np.insert(results["qstat"], 0, np.nan)
        pvalues_padded = np.insert(results["pvalues"], 0, np.nan)
        
        df = pd.DataFrame({
            "Lag": lags_index,
            "ACF": results["acf_values"],
            "ACF_Lower_CI": results["acf_ci"][:, 0],
            "ACF_Upper_CI": results["acf_ci"][:, 1],
            "PACF": results["pacf_values"],
            "PACF_Lower_CI": results["pacf_ci"][:, 0],
            "PACF_Upper_CI": results["pacf_ci"][:, 1],
            "Ljung-Box Q-Stat": qstat_padded,
            "Ljung-Box p-value": pvalues_padded
        })
        
        df.set_index("Lag", inplace=True)
        df.to_csv(self.file_path_stats)
        print(f"Estatísticas da Questão 1 salvas em: {self.file_path_stats}")

    def _interpret_results(self, results: dict) -> str:
        acf_values = results["acf_values"]
        acf_ci = results["acf_ci"]
        
        interpretation = "Questão 1: Interpretação da Análise de Autocorrelação:\n"
        
        # 1. Sazonalidade
        interpretation += "1. Análise de Sazonalidade:\n"
        seasonal_lags = [i * self.freq for i in range(1, (len(acf_values) // self.freq) + 1)]
        seasonal_peaks = []
        for lag in seasonal_lags:
            if lag < len(acf_values):
                # Verifica se é significativo (0 fora do intervalo de confiança)
                lower = acf_ci[lag, 0]
                upper = acf_ci[lag, 1]
                if not (lower <= 0 <= upper):
                    seasonal_peaks.append(lag)
        
        if seasonal_peaks:
            interpretation += f"* Observam-se picos significativos nas defasagens sazonais: {seasonal_peaks}.\n"
            interpretation += "* O padrão sugere a presença de sazonalidade na série.\n"
        else:
            interpretation += "* Não foram observados picos significativos nas defasagens sazonais esperadas.\n"
            interpretation += "* O padrão não sugere sazonalidade forte na frequência analisada.\n"
        
        interpretation += "\n"

        # 2. Dependência Temporal / Persistência
        interpretation += "2. Análise de Dependência Temporal (Persistência):\n"
        # Verifica taxa de decaimento. Heurística: conta quantas defasagens iniciais são significativas.
        significant_lags = 0
        for i in range(1, len(acf_values)):
            lower = acf_ci[i, 0]
            upper = acf_ci[i, 1]
            if not (lower <= 0 <= upper):
                significant_lags += 1
            else:
                # Se encontrar um não significativo, paramos a contagem de persistência contínua
                break 
        
        interpretation += f"* A autocorrelação permanece significativa continuamente para as primeiras {significant_lags} defasagens.\n"
        
        # Se a persistência for longa (ex: maior que meio ciclo ou maior que um valor arbitrário como 5)
        if significant_lags > 5: 
            interpretation += "* Há indícios de alta persistência (autocorrelação lenta), o que pode sugerir não-estacionariedade ou memória longa.\n"
        else:
            interpretation += "* A autocorrelação decai rapidamente, sugerindo uma série estacionária ou com dependência de curto prazo.\n"

        return interpretation

    # executa a questão 1
    def run(self):
        results = self._calculate_autocorrelation()
        self._plot_acf_pacf()
        self._save_stats(results)
        
        interpretation = self._interpret_results(results)
        file_path_interpretation = os.path.join(self.output_dir, "q1_interpretation.txt")
        with open(file_path_interpretation, 'w') as f:
            f.write(interpretation)
        print(f"Interpretação da Questão 1 salva em: {file_path_interpretation}")
        


