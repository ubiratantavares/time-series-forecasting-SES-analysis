import os
import pandas as pd

from model.questao1 import Questao1
from model.questao2 import Questao2
from model.questao3 import Questao3
from model.questao4 import Questao4
from model.questao5 import Questao5
from model.relatorio import Relatorio

import json

"""
Classe responsável por orquestrar todo o fluxo de trabalho da lista prática.
"""

class Controller:

    def __init__(self, serie: pd.Series, freq: int, h: int = 12, output_dir: str = "output/"):
        self.serie = serie
        self.freq = freq
        self.h = h
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        # Salvar configurações para uso no Relatório
        config = {"freq": self.freq, "h": self.h}
        with open(os.path.join(self.output_dir, "config.json"), "w") as f:
            json.dump(config, f)
            
        self.questao1 = Questao1(self.serie, self.freq, self.output_dir)
        self.questao2 = Questao2(self.serie, self.output_dir)
        self.questao3 = Questao3(self.serie, self.h, self.output_dir)
        self.questao4 = Questao4(self.serie, self.output_dir)
        self.questao5 = Questao5(self.serie, self.h, self.output_dir)
        self.relatorio = Relatorio(self.output_dir)

    # executa a Questão 1: Período/Autocorrelação
    def _run_questao1(self):
        self.questao1.run()

    # executa a Questão 2: Estacionaridade (ADF e KPSS)
    def _run_questao2(self):
        self.questao2.run()

    # executa a Questão 3: Previsão SES
    def _run_questao3(self):
        self.questao3.run()

    # executa a Questão 4: Diagnóstico de Outliers
    def _run_questao4(self):
        self.questao4.run()

    # executa a Questão 5: Conclusão Geral
    def _run_questao5(self):
        self.questao5.run()

    # gera o Relatório Final
    def _run_relatorio(self):
        self.relatorio.run()

    def run(self):
        self._run_questao1()
        self._run_questao2()
        self._run_questao3()
        self._run_questao4()
        self._run_questao5()
        self._run_relatorio()