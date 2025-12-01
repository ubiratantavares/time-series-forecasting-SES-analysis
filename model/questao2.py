import os
import pandas as pd
import warnings
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tools.sm_exceptions import InterpolationWarning
from abstract.analysis import Analysis

"""
Classe responsável por responder aos objetivos da Questão 2.
"""
class Questao2(Analysis):

    def __init__(self, serie: pd.Series, output_dir: str):
        self.serie = serie.dropna()
        self.output_dir = output_dir
        self.file_path_results = os.path.join(self.output_dir, "q2_stationarity_results.csv")
        # self.file_path_interpretation = os.path.join(self.output_dir, "q2_interpretation.txt") # Removed

    def _perform_adf_test(self) -> dict:
        """
        Executa o teste Augmented Dickey-Fuller (ADF).
        H0: A série possui uma raiz unitária (não é estacionária).
        H1: A série não possui raiz unitária (é estacionária).
        """
        result = adfuller(self.serie, autolag='AIC')
        return {
            'Test Statistic': result[0],
            'p-value': result[1],
            'Lags Used': result[2],
            'Number of Observations Used': result[3],
            'Critical Values': result[4],
            'IC Best': result[5] if len(result) > 5 else None
        }

    def _perform_kpss_test(self) -> dict:
        """
        Executa o teste Kwiatkowski-Phillips-Schmidt-Shin (KPSS).
        H0: A série é estacionária em torno de uma média determinística (ou tendência).
        H1: A série possui uma raiz unitária (não é estacionária).
        """
        # 'c' : stationarity around level (default)
        # 'ct': stationarity around trend
        # Vamos testar ambos ou assumir 'c' inicialmente? 
        # O enunciado menciona "tendência/raiz unitária", então 'ct' pode ser relevante.
        # Mas geralmente começa-se com 'c' (level). Vamos fazer 'c' por padrão.
        
        # Nota: statsmodels avisa sobre 'nlags'="auto" ou "legacy".
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=InterpolationWarning)
            result = kpss(self.serie, regression='c', nlags='auto')
            
        return {
            'Test Statistic': result[0],
            'p-value': result[1],
            'Lags Used': result[2],
            'Critical Values': result[3]
        }

    def run(self):
        adf_results = self._perform_adf_test()
        kpss_results = self._perform_kpss_test()
        
        # Salvar resultados numéricos
        results_list = []

        # ADF
        for k, v in adf_results.items():
            if k == 'Critical Values':
                for cv_k, cv_v in v.items():
                    results_list.append({'Test': 'ADF', 'Metric': f'Critical Value {cv_k}', 'Value': cv_v})
            else:
                results_list.append({'Test': 'ADF', 'Metric': k, 'Value': v})
        
        # KPSS
        for k, v in kpss_results.items():
            if k == 'Critical Values':
                for cv_k, cv_v in v.items():
                    results_list.append({'Test': 'KPSS', 'Metric': f'Critical Value {cv_k}', 'Value': cv_v})
            else:
                results_list.append({'Test': 'KPSS', 'Metric': k, 'Value': v})

        df_results = pd.DataFrame(results_list)
        df_results.to_csv(self.file_path_results, index=False)
        print(f"Resultados numéricos salvos em: {self.file_path_results}")



