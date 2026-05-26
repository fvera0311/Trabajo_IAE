import dask.dataframe as dd
import pandas as pd
import numpy as np
from pathlib import Path


class DataProcessor:

    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.ddf = None
        self.df = None
        
    def load_data(self, use_dask: bool = True):
        """Carga datos usando Dask o Pandas"""
        if use_dask:
            # Importación con Dask para procesamiento paralelo
            self.ddf = dd.read_csv(self.data_path, dtype= {'age':'float64'})
            self.df = self.ddf.compute()
        else:
            self.df = pd.read_csv(self.data_path)
        
        print(f"✅ Datos cargados: {len(self.df)} registros, {len(self.df.columns)} columnas")
        return self.df
    
# Filtro
    def filter_data(self, conditions: dict = None):
        """Filtra datos según condiciones especificadas"""
        if conditions is None:
            conditions = {
                'age': lambda x: x > 60,
                'diabetes': lambda x: x == 1
            }
        
        filtered_df = self.df.copy()
        
        for column, condition in conditions.items():
            if column in filtered_df.columns:
                filtered_df = filtered_df[filtered_df[column].apply(condition)]
        
        print(f"✅ Filtrado aplicado: {len(filtered_df)} registros")
        return filtered_df
    
# estadisticas por variables
    def aggregate_data(self):
        """Realiza agregaciones estadísticas sobre los datos"""
        aggregations = {}
        

        death_agg = self.df.groupby('DEATH_EVENT').agg({
            'age': ['mean', 'std', 'min', 'max'],
            'ejection_fraction': ['mean', 'std'],
            'serum_creatinine': ['mean', 'std'],
            'time': ['mean', 'std']
        }).round(2)
        aggregations['por_muerte'] = death_agg
        

        diabetes_agg = self.df.groupby('diabetes').agg({
            'age': 'mean',
            'DEATH_EVENT': 'sum',
            'ejection_fraction': 'mean'
        }).round(2)
        aggregations['por_diabetes'] = diabetes_agg
        
   
        gender_agg = self.df.groupby('sex').agg({
            'age': 'mean',
            'DEATH_EVENT': ['sum', 'mean'],
            'high_blood_pressure': 'sum'
        }).round(2)
        aggregations['por_genero'] = gender_agg
        
        print(f"✅ Agregaciones completadas: {len(aggregations)} grupos")
        return aggregations
    
    # Juntar valores
    def map_transformations(self):
        """Aplica funciones de transformación elemento a elemento"""
        df_mapped = self.df.copy()
        
        # Mapeo 1: Por edad
        def categorize_age(age):
            if age < 50:
                return 'Joven'
            elif age < 70:
                return 'Mediana Edad'
            else:
                return 'Anciano'
        
        df_mapped['age_category'] = df_mapped['age'].apply(categorize_age)
        
        # Mapeo 2: Según fracción de eyección
        def ejection_risk(ef):
            if ef < 30:
                return 'Alto Riesgo'
            elif ef < 50:
                return 'Riesgo Moderado'
            else:
                return 'Bajo Riesgo'
        
        df_mapped['ejection_risk'] = df_mapped['ejection_fraction'].apply(ejection_risk)
        
        continuous_vars = ['age', 'creatinine_phosphokinase', 'ejection_fraction', 
                          'serum_creatinine', 'serum_sodium', 'platelets']
        
        for var in continuous_vars:
            min_val = df_mapped[var].min()
            max_val = df_mapped[var].max()
            df_mapped[f'{var}_normalized'] = (df_mapped[var] - min_val) / (max_val - min_val)
        

        df_mapped['comorbidity_score'] = (
            df_mapped['anaemia'] + 
            df_mapped['diabetes'] + 
            df_mapped['high_blood_pressure'] + 
            df_mapped['smoking']
        )
        
        print(f"✅ Transformaciones completadas")
        return df_mapped
    

    def sort_data(self, criteria: list = None, ascending: bool = False):
        """Ordena datos según múltiples criterios"""
        if criteria is None:
            criteria = ['DEATH_EVENT', 'age', 'ejection_fraction']
        
        sorted_df = self.df.sort_values(
            by=criteria, 
            ascending=ascending
        ).reset_index(drop=True)
        
        # Ordenaciones adicionales
        ordenaciones = {}
        
        # Ordenar por riesgo
        df_risk = self.df.copy()
        df_risk['risk_score'] = (
            (100 - df_risk['ejection_fraction']) * 0.5 + 
            df_risk['serum_creatinine'] * 20
        )
        ordenaciones['por_riesgo'] = df_risk.sort_values('risk_score', ascending=False)
        
        print(f"✅ Ordenación completada")
        return sorted_df, ordenaciones
    
    def prepare_for_modeling(self):
        """Prepara datos para modelado"""
        X = self.df.drop(['DEATH_EVENT','time'], axis=1)
        y = self.df['DEATH_EVENT']
        return X, y
