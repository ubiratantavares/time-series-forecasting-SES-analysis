import os
import pandas as pd
from model.data import Data
from controller.controller import Controller

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "dataset", "daily-total-female-births.csv")
    
    # Carregar dados
    serie = pd.read_csv(file_path, header=0, index_col=0, parse_dates=True).squeeze()
    serie.index.freq = 'D' # Define frequência diária para evitar ValueWarning
    
    # define a frequência para capturar sazonalidade semanal.
    freq = 7

    # define o horizonte de previsão h=7 (uma semana)
    h = 7
    
    # executa o controlador
    controller = Controller(serie, freq, h)
    controller.run()

if __name__ == "__main__":
    main()