import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


class DataVisualizer:
    """Clase para crear visualizaciones exploratorias y explicativas"""
    
    NOMBRES_ESPANOL = {
        'age': 'Edad',
        'anaemia': 'Anemia',
        'creatinine_phosphokinase': 'CPK',
        'diabetes': 'Diabetes',
        'ejection_fraction': 'Frac. Eyección',
        'high_blood_pressure': 'Hipertensión',
        'platelets': 'Plaquetas',
        'serum_creatinine': 'Creatinina',
        'serum_sodium': 'Sodio',
        'sex': 'Sexo',
        'smoking': 'Fumador',
        'time' : 'Observación (días)',
        'DEATH_EVENT': 'Fallecimiento'
    }
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        sns.set_theme(style="whitegrid", palette="muted")
        plt.rcParams['figure.figsize'] = (12, 8)
        
    def plot_target_distribution(self):
        """Distribución de la variable objetivo"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        death_counts = self.df['DEATH_EVENT'].value_counts().sort_index()
        
        color_map = {0: '#2ecc71', 1: '#e74c3c'}
        label_map = {0: 'Sobrevivió', 1: 'Falleció'}
        
        colores = [color_map[idx] for idx in death_counts.index]
        etiquetas = [label_map[idx] for idx in death_counts.index]
        
        # Gráfico de Barras
        axes[0].bar(etiquetas, death_counts.values, 
                    color=colores, alpha=0.7, edgecolor='black')
        axes[0].set_ylabel('Número de Pacientes')
        axes[0].set_title('Distribución de Mortalidad', fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)
        
        # Diagrama de Sectores
        axes[1].pie(death_counts.values, labels=etiquetas,
                   autopct='%1.1f%%', colors=colores,
                   startangle=90)
        axes[1].set_title('Proporción de Mortalidad', fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def plot_age_distribution(self):
        """Distribución de edad por estado vital"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        for death_event in [0, 1]:
            subset = self.df[self.df['DEATH_EVENT'] == death_event]['age']
            label = 'Sobrevivió' if death_event == 0 else 'Falleció'
            color = '#2ecc71' if death_event == 0 else '#e74c3c'
            axes[0].hist(subset, bins=20, alpha=0.6, label=label, color=color)
        
        axes[0].set_xlabel('Edad (años)')
        axes[0].set_ylabel('Frecuencia')
        axes[0].set_title('Distribución de Edad', fontweight='bold')
        axes[0].legend()
        
        # Boxplot con nombres en español
        df_temp = self.df.copy()
        df_temp['Estado'] = df_temp['DEATH_EVENT'].map({0: 'Sobrevivió', 1: 'Falleció'})
        
        sns.boxplot(data=df_temp, x='Estado', y='age', ax=axes[1],
                   palette={'Sobrevivió': '#2ecc71', 'Falleció': '#e74c3c'})
        axes[1].set_xlabel('Estado Vital')
        axes[1].set_ylabel('Edad (años)')
        axes[1].set_title('Edad por Estado Vital', fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def plot_correlation_matrix(self):
        """Matriz de correlación"""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        corr_matrix = self.df[numeric_cols].corr()
        
        # Traducir nombres SOLO para visualización
        corr_matrix = corr_matrix.rename(
            columns=self.NOMBRES_ESPANOL, 
            index=self.NOMBRES_ESPANOL
        )
        
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlGn_r',
                   center=0, square=True, linewidths=1, ax=ax,
                   cbar_kws={'label': 'Correlación'})
        
        ax.set_title('Matriz de Correlación', fontweight='bold', pad=20)
        plt.tight_layout()
        return fig
    
    def plot_clinical_variables(self):
        """Variables clínicas principales"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        clinical_vars = [
            ('ejection_fraction', 'Fracción de Eyección (%)'),
            ('serum_creatinine', 'Creatinina Sérica (mg/dL)'),
            ('serum_sodium', 'Sodio Sérico (mEq/L)'),
            ('creatinine_phosphokinase', 'CPK (mcg/L)')
        ]
        
        for idx, (var, label) in enumerate(clinical_vars):
            ax = axes[idx // 2, idx % 2]
            
            for death_event in [0, 1]:
                subset = self.df[self.df['DEATH_EVENT'] == death_event][var]
                label_event = 'Sobrevivió' if death_event == 0 else 'Falleció'
                color = '#2ecc71' if death_event == 0 else '#e74c3c'
                ax.hist(subset, bins=20, alpha=0.6, label=label_event, color=color)
            
            ax.set_xlabel(label)
            ax.set_ylabel('Frecuencia')
            ax.set_title(f'Distribución de {label.split("(")[0].strip()}', fontsize=12, fontweight='bold')
            ax.legend()
        
        plt.tight_layout()
        return fig
    
    def plot_comorbidities(self):
        """Análisis de comorbilidades"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        comorbidities = [
            ('diabetes', 'Diabetes'),
            ('anaemia', 'Anemia'),
            ('high_blood_pressure', 'Hipertensión'),
            ('smoking', 'Fumador')
        ]
        
        for idx, (var, label) in enumerate(comorbidities):
            ax = axes[idx // 2, idx % 2]
            
            contingency = pd.crosstab(
                self.df[var], 
                self.df['DEATH_EVENT'],
                normalize='index'
            ) * 100
            
            contingency.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'], alpha=0.7)
            ax.set_xlabel(label)
            ax.set_ylabel('Porcentaje (%)')
            ax.set_title(f'Mortalidad por {label}', fontweight='bold')
            ax.set_xticklabels(['No', 'Sí'], rotation=0)
            ax.legend(['Sobrevivió', 'Falleció'])
        
        plt.tight_layout()
        return fig
