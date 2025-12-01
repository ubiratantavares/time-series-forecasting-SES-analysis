import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from abstract.analysis import Analysis

"""
Classe responsável por responder aos objetivos da Questão 3.
"""
class Questao3(Analysis):

    def __init__(self, serie: pd.Series, h: int, output_dir: str):
        self.serie = serie.dropna()
        self.h = h
        self.output_dir = output_dir
        self.file_path_metrics = os.path.join(self.output_dir, "q3_metrics.csv")
        self.file_path_plot = os.path.join(self.output_dir, "q3_forecast_plot.png")
        self.file_path_interpretation = os.path.join(self.output_dir, "q3_interpretation.txt")

    def _split_data(self):
        """
        Divide os dados em treino e teste.
        O conjunto de teste terá o tamanho de h (horizonte de previsão).
        """
        train = self.serie.iloc[:-self.h]
        test = self.serie.iloc[-self.h:]
        return train, test

    def _fit_predict(self, train: pd.Series):
        """
        Ajusta o modelo SES nos dados de treino e faz a previsão.
        """
        # Ajusta o modelo SES. 
        # initialization_method='estimated' estima o valor inicial.
        model = SimpleExpSmoothing(train, initialization_method="estimated").fit()
        
        # Previsão h passos à frente
        forecast = model.forecast(self.h)
        
        return model, forecast

    def _calculate_metrics(self, test: pd.Series, forecast: pd.Series, model) -> dict:
        """
        Calcula métricas de acurácia: RMSE, MAE, MAPE e extrai Alpha.
        """
        rmse = np.sqrt(mean_squared_error(test, forecast))
        mae = mean_absolute_error(test, forecast)
        mape = mean_absolute_percentage_error(test, forecast) * 100 # Em porcentagem
        alpha = model.params['smoothing_level']
        
        return {
            "RMSE": rmse,
            "MAE": mae,
            "MAPE": mape,
            "Alpha": alpha
        }

    def _plot_results(self, train: pd.Series, test: pd.Series, forecast: pd.Series):
        """
        Gera gráfico comparando Treino, Teste e Previsão.
        """
        plt.figure(figsize=(12, 6))
        plt.plot(train.index, train, label='Treino')
        plt.plot(test.index, test, label='Teste (Real)', color='green')
        plt.plot(forecast.index, forecast, label='Previsão SES', color='red', linestyle='--')
        plt.title(f'Previsão SES - Horizonte h={self.h}')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.file_path_plot)
        plt.close()
        print(f"Gráfico de previsão salvo em: {self.file_path_plot}")

    def _interpret_results(self, model, metrics: dict) -> str:
        """
        Interpreta o valor de alpha e a acurácia.
        """
        alpha = model.params['smoothing_level']
        rmse = metrics['RMSE']
        mape = metrics['MAPE']
        
        interpretation = "Questão 3: Interpretação dos Resultados SES:\n\n"
        
        # 1. Parâmetro de Suavização (Alpha)
        interpretation += f"1. Parâmetro de Suavização (Alpha): {alpha:.4f}\n"
        if alpha > 0.8:
            interpretation += "* O valor de alpha é alto (próximo de 1).\n"
            interpretation += "* Isso indica que o modelo dá muito peso às observações mais recentes.\n"
            interpretation += "* A previsão reage rapidamente a mudanças recentes no nível da série (pouca suavização).\n"
        elif alpha < 0.2:
            interpretation += "* O valor de alpha é baixo (próximo de 0).\n"
            interpretation += "* Isso indica que o modelo dá peso similar a todo o histórico (memória longa).\n"
            interpretation += "* A previsão é muito suave e reage lentamente a mudanças recentes.\n"
        else:
            interpretation += "* O valor de alpha é intermediário.\n"
            interpretation += "* O modelo equilibra a importância do histórico recente e passado.\n"
        
        interpretation += "\n"
        
        # 2. Acurácia (MAPE e RMSE)
        interpretation += "2. O que indicaram os resultados do MAPE e RMSE?\n"
        interpretation += f"* MAPE: {mape:.2f}%\n"
        interpretation += f"* RMSE: {rmse:.4f}\n"
        
        if mape < 10:
            interpretation += "* O MAPE abaixo de 10% indica uma acurácia excelente.\n"
        elif mape < 20:
            interpretation += "* O MAPE entre 10% e 20% indica uma boa acurácia.\n"
        elif mape < 50:
            interpretation += "* O MAPE entre 20% e 50% indica uma acurácia razoável.\n"
        else:
            interpretation += "* O MAPE acima de 50% indica uma acurácia baixa.\n"
            
        interpretation += "* O RMSE fornece uma estimativa do desvio padrão dos erros de previsão na mesma escala dos dados.\n"
        interpretation += "\n"

        # 3. Adequação do SES
        interpretation += "3. O método SES é adequado à série?\n"
        interpretation += "* O SES (Suavização Exponencial Simples) é ideal para séries SEM tendência e SEM sazonalidade claras,\n"
        interpretation += "* pois projeta um nível constante (flat forecast).\n"
        
        # Heurística simples baseada no alpha e erro
        if alpha > 0.9:
             interpretation += "* O alpha muito alto sugere que o modelo está tentando 'correr atrás' dos dados, possivelmente indicando uma tendência não modelada (Naive method behavior).\n"
        
        if mape > 20:
             interpretation += "* O erro elevado (MAPE > 20%) pode sugerir que o modelo SES é insuficiente para capturar a dinâmica da série.\n"
             interpretation += "* Se a série apresentar tendência ou sazonalidade (verificar Q1/Q2), métodos como Holt (tendência) ou Holt-Winters (sazonalidade) seriam mais adequados.\n"
        else:
             interpretation += "* Dado o erro relativamente baixo, o SES parece fornecer uma aproximação razoável para o horizonte de curto prazo,\n"
             interpretation += "* embora deva-se ter cautela se houver evidências de tendência/sazonalidade nos testes anteriores.\n"

        return interpretation

    def run(self):
        train, test = self._split_data()
        model, forecast = self._fit_predict(train)
        metrics = self._calculate_metrics(test, forecast, model)
        
        self._plot_results(train, test, forecast)
        
        # Salvar métricas
        df_metrics = pd.DataFrame([metrics])
        df_metrics.to_csv(self.file_path_metrics, index=False)
        print(f"Métricas salvas em: {self.file_path_metrics}")
        
        # Salvar interpretação
        interpretation = self._interpret_results(model, metrics)
        with open(self.file_path_interpretation, 'w') as f:
            f.write(interpretation)
        print(f"Interpretação salva em: {self.file_path_interpretation}")
