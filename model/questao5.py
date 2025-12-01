import os
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from abstract.analysis import Analysis

"""
Classe responsável por responder aos objetivos da Questão 5.
"""
class Questao5(Analysis):

    def __init__(self, serie: pd.Series, h: int, output_dir: str):
        self.serie = serie.dropna()
        self.h = h
        self.output_dir = output_dir
        self.file_path_conclusion = os.path.join(self.output_dir, "q5_general_conclusion.txt")

    def _fit_evaluate_model(self):
        """
        Re-ajusta o modelo SES e calcula métricas para embasar a conclusão.
        """
        # Divisão Treino/Teste
        train = self.serie.iloc[:-self.h]
        test = self.serie.iloc[-self.h:]
        
        # Ajuste
        model = SimpleExpSmoothing(train, initialization_method="estimated").fit()
        forecast = model.forecast(self.h)
        
        # Métricas
        rmse = np.sqrt(mean_squared_error(test, forecast))
        mape = mean_absolute_percentage_error(test, forecast) * 100
        alpha = model.params['smoothing_level']
        
        return alpha, rmse, mape

    def _generate_conclusion(self, alpha: float, rmse: float, mape: float) -> str:
        """
        Gera o texto da conclusão geral.
        """
        conclusion = "Questão 5: Conclusão Geral sobre o Modelo Estimado:\n\n"
        
        conclusion += "Com base nas análises realizadas (Sazonalidade, Estacionariedade, Previsão SES e Outliers), conclui-se que:\n\n"
        
        # 1. Desempenho do Modelo (Acurácia)
        conclusion += "1. Desempenho do Modelo (Acurácia):\n"
        conclusion += f"* O modelo SES apresentou um MAPE de {mape:.2f}% e um RMSE de {rmse:.4f}.\n"
        if mape < 20:
            conclusion += "* O erro percentual é relativamente baixo, indicando que o modelo consegue capturar o nível da série com razoável precisão a curto prazo.\n"
        else:
            conclusion += "* O erro percentual é elevado, sugerindo que o modelo tem dificuldades em prever a série com precisão.\n"
        conclusion += "\n"
        
        # 2. Adequação Teórica (Alpha e Suposições)
        conclusion += "2. Adequação do Método SES:\n"
        conclusion += f"* O parâmetro de suavização (alpha) estimado foi de {alpha:.4f}.\n"
        if alpha < 0.2:
            conclusion += "* Um alpha baixo indica que a série possui uma memória longa e o nível muda lentamente.\n"
            conclusion += "* O SES é adequado para séries estacionárias ou com mudanças de nível lentas e sem tendência/sazonalidade determinísticas.\n"
        elif alpha > 0.8:
            conclusion += "* Um alpha alto sugere que a previsão segue muito os dados recentes (quase um Naive).\n"
            conclusion += "* Isso pode ser um sintoma de que o modelo está tentando compensar uma tendência ou sazonalidade não modelada.\n"
        
        conclusion += "* IMPORTANTE: O SES projeta uma previsão constante (flat). Se as análises anteriores (Q1/Q2) indicaram sazonalidade ou tendência,\n"
        conclusion += "* o SES é teoricamente INSUFICIENTE para previsões de longo prazo, pois ignorará esses componentes estruturais.\n"
        conclusion += "\n"
        
        # 3. Confiabilidade e Robustez
        conclusion += "3. Confiabilidade e Robustez:\n"
        conclusion += "* A presença de outliers (diagnosticada na Q4) deve ser considerada. Se houver outliers recentes, a previsão do SES (que depende do nível final) pode ser enviesada.\n"
        conclusion += "* A simplicidade do SES é uma vantagem para robustez (menos parâmetros para estimar), mas uma desvantagem para capturar dinâmicas complexas.\n"
        conclusion += "\n"
        
        # 4. Veredito Final
        conclusion += "4. Veredito Final:\n"
        if mape < 20 and alpha < 0.8:
            conclusion += "* O modelo SES é ACEITÁVEL para previsões de curtíssimo prazo (h pequeno), dada sua simplicidade e acurácia razoável neste horizonte.\n"
            conclusion += "* No entanto, para horizontes maiores ou se a sazonalidade for confirmada como relevante, recomenda-se testar modelos mais completos (ex: Holt-Winters ou SARIMA).\n"
        else:
            conclusion += "* O modelo SES apresenta LIMITAÇÕES CLARAS para esta série.\n"
            conclusion += "* Recomenda-se fortemente o uso de modelos que incorporem tendência e/ou sazonalidade para melhorar a capacidade preditiva.\n"

        return conclusion

    def run(self):
        alpha, rmse, mape = self._fit_evaluate_model()
        conclusion = self._generate_conclusion(alpha, rmse, mape)
        
        with open(self.file_path_conclusion, 'w') as f:
            f.write(conclusion)
        print(f"Conclusão geral salva em: {self.file_path_conclusion}")

