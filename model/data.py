import pandas as pd

class Data:
    """
    Classe responsável por preparar os dados brutos.
    """

    def __init__(self, file_path):
        self.file_path = file_path

    def load(self, col_name='Births'):
        return pd.read_csv(self.file_path, header=0, index_col=0, parse_dates=True)
