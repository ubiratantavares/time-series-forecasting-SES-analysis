import pandas as pd

"""
Classe responsável por preparar os dados brutos.
"""
class DataProcessor:

    def __init__(self, file_path, freq, split_h):
        self.file_path = file_path
        self.freq = freq
        self.split_h = split_h

    def load(self, col_name='Births'):
        """
        Carrega a série diária e a ajusta para frequência mensal (MS - Month Start)
        para cumprir o requisito da lista (frequência 12).
        """
        return pd.read_csv(self.file_path, header=0, index_col=0, parse_dates=True)
