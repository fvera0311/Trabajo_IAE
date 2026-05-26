import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / 'src'))

from src.data_processing import DataProcessor
from src.visualization import DataVisualizer
from src.models import HeartFailureModels
from src.r_integration import RPythonIntegration

st.set_page_config(
    page_title="Predicción Insuficiencia Cardíaca",
    page_icon="❤️",
    layout="wide"
)

st.markdown("""<style>.main-header {font-size: 2.5rem; font-weight: bold; color: #e74c3c; text-align: center;}</style>""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    processor = DataProcessor('data/heart_failure_dataset.csv')
    df = processor.load_data(use_dask=True)
    return processor, df


@st.cache_resource
def train_models(_processor):  # FIX: Guion bajo para evitar hashing
    X, y = _processor.prepare_for_modeling()
    models = HeartFailureModels(random_state=42)
    models.prepare_data(X, y, test_size=0.2, use_smote=True)
    
    models.train_random_forest()
    models.train_logistic_regression()
    models.train_gradient_boosting()
    models.train_svm()
    models.train_deep_learning(epochs=50, batch_size=16)
    
    return models


def main():
    st.markdown('<p class="main-header"> Predicción de Mortalidad por Insuficiencia Cardíaca</p>', 
                unsafe_allow_html=True)
    
    st.sidebar.title("📊 Navegación")
    page = st.sidebar.radio(
        "Selecciona:",
        ["🏠 Inicio", "🔬 Tratamiento", "📊 Visualizaciones", 
         "🤖 Modelos ML", "🧠 Deep Learning", "📉 Comparación", "🎯 Predictor"]
    )
    
    processor, df = load_data()
    
    if page == "🏠 Inicio":
        st.markdown("## 📋 Resumen Clínico del Estudio")
        
        # Métricas principales
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Pacientes", len(df))
        col2.metric("Pacientes Fallecidos", df['DEATH_EVENT'].sum())
        col3.metric("Tasa de Mortalidad", f"{(df['DEATH_EVENT'].mean()*100):.1f}%")
        col4.metric("Edad Promedio", f"{df['age'].mean():.0f} años")
        
        st.markdown("---")
        
        # tabla con nombres en español
        st.markdown("### Muestra de Pacientes del Estudio")
        
        df_display = df.head(10).copy()
        df_display = df_display.rename(columns={
            'age': 'Edad',
            'anaemia': 'Anemia',
            'creatinine_phosphokinase': 'CPK',
            'diabetes': 'Diabetes',
            'ejection_fraction': 'Fracción Eyección (%)',
            'high_blood_pressure': 'Hipertensión',
            'platelets': 'Plaquetas',
            'serum_creatinine': 'Creatinina',
            'serum_sodium': 'Sodio',
            'sex': 'Sexo',
            'smoking': 'Fumador',
            'time' : 'Observación (días)',
            'DEATH_EVENT': 'Fallecimiento'
        })

        binary_cols = ['Anemia', 'Diabetes', 'Hipertensión', 'Fumador', 'Fallecimiento']
        for col in binary_cols:
            df_display[col] = df_display[col].map({0: 'No', 1: 'Sí'})

        df_display['Sexo'] = df_display['Sexo'].map({0: 'Mujer', 1: 'Hombre'})

        df_display['Edad'] = df_display['Edad'].astype(int)
        df_display['CPK'] = df_display['CPK'].astype(int)
        df_display['Plaquetas'] = df_display['Plaquetas'].astype(int)
        df_display['Creatinina'] = df_display['Creatinina'].round(2)
        df_display['Sodio'] = df_display['Sodio'].astype(int)

        st.dataframe(
            df_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Edad": st.column_config.NumberColumn("Edad (años)", format="%d"),
                "Fracción Eyección (%)": st.column_config.NumberColumn("Fracción Eyección", format="%d%%"),
                "Creatinina": st.column_config.NumberColumn("Creatinina (mg/dL)", format="%.2f"),
                "Sodio": st.column_config.NumberColumn("Sodio (mEq/L)", format="%d"),
                "CPK": st.column_config.NumberColumn("CPK (mcg/L)", format="%d"),
                "Plaquetas": st.column_config.NumberColumn("Plaquetas", format="%d"),
            }
        )

        
        st.markdown("---")

        st.markdown("### Indicadores Clínicos Clave")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Fracción de Eyección**")
            st.metric("Promedio", f"{df['ejection_fraction'].mean():.1f}%")
            st.metric("Pacientes < 30% (Alto Riesgo)", 
                     f"{(df['ejection_fraction'] < 30).sum()} ({(df['ejection_fraction'] < 30).mean()*100:.1f}%)")
        
        with col2:
            st.markdown("**Creatinina Sérica**")
            st.metric("Promedio", f"{df['serum_creatinine'].mean():.2f} mg/dL")
            st.metric("Pacientes > 2.0 mg/dL (Elevada)", 
                     f"{(df['serum_creatinine'] > 2.0).sum()} ({(df['serum_creatinine'] > 2.0).mean()*100:.1f}%)")
        
        with col3:
            st.markdown("**Comorbilidades**")
            st.metric("Pacientes Diabéticos", 
                     f"{df['diabetes'].sum()} ({df['diabetes'].mean()*100:.1f}%)")
            st.metric("Pacientes Hipertensos", 
                     f"{df['high_blood_pressure'].sum()} ({df['high_blood_pressure'].mean()*100:.1f}%)")
    
    elif page == "🔬 Tratamiento":
        st.markdown("## Manipulación de datos")
        
        traducciones_dask = {
            'age': 'Edad', 'anaemia': 'Anemia', 'creatinine_phosphokinase': 'CPK',
            'diabetes': 'Diabetes', 'ejection_fraction': 'Fracción Eyección (%)',
            'high_blood_pressure': 'Hipertensión', 'platelets': 'Plaquetas',
            'serum_creatinine': 'Creatinina', 'serum_sodium': 'Sodio',
            'sex': 'Sexo', 'smoking': 'Fumador', 'time': 'Observación (días)',
            'DEATH_EVENT': 'Fallecimiento', 'age_category': 'Categoría Edad',
            'ejection_risk': 'Riesgo Eyección'
        }

        st.markdown("### Filtrado")
        
        age_range = st.slider("Filtro de edad:", 40, 95, (40, 95))
        age_min, age_max = age_range

        st.markdown("**Filtrar por condiciones:**")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            anaemia_filter = st.checkbox("Sólo anémicos")
        with col2:
            diabetes_filter = st.checkbox("Sólo diabéticos")
        with col3:
            hypertension_filter = st.checkbox("Sólo hipertensos")
        with col4:
            smoking_filter = st.checkbox("Sólo fumadores")

        conditions = {'age': lambda x: (x >= age_min) & (x <= age_max)}
        
        if anaemia_filter:
            conditions['anaemia'] = lambda x: x == 1
        if diabetes_filter:
            conditions['diabetes'] = lambda x: x == 1
        if hypertension_filter:
            conditions['high_blood_pressure'] = lambda x: x == 1
        if smoking_filter:
            conditions['smoking'] = lambda x: x == 1
            
        filtered = processor.filter_data(conditions)
        active_filters = sum([anaemia_filter, diabetes_filter, hypertension_filter, smoking_filter])
        
        if active_filters > 0:
            st.info(f"🔍 {active_filters} filtro(s) de comorbilidad activo(s)")
        
        st.write(f"**Pacientes encontrados:** {len(filtered)} registros")
        
        st.dataframe(filtered.rename(columns=traducciones_dask), use_container_width=True)
        
        st.markdown("---")
        
        st.markdown("### Agregación")
        aggs = processor.aggregate_data()
        df_aggs = aggs['por_muerte'].reset_index().rename(columns=traducciones_dask)
        st.dataframe(df_aggs, use_container_width=True)
        
        st.markdown("###  Mapeo")
        mapped = processor.map_transformations()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📌 Transformación 1: Categorías de Edad**")
            st.dataframe(mapped[['age', 'age_category']].head(10).rename(columns=traducciones_dask))
            age_dist = mapped['age_category'].value_counts()
            st.write("**Distribución:**")
            st.write(f"- Anciano: {age_dist.get('Anciano', 0)} pacientes")
            st.write(f"- Mediana Edad: {age_dist.get('Mediana Edad', 0)} pacientes")
            st.write(f"- Joven: {age_dist.get('Joven', 0)} pacientes")
        
        with col2:
            st.write("**📌 Transformación 2: Riesgo por Eyección**")
            st.dataframe(mapped[['ejection_fraction', 'ejection_risk']].head(10).rename(columns=traducciones_dask))
            risk_dist = mapped['ejection_risk'].value_counts()
            st.write("**Distribución:**")
            st.write(f"- Alto Riesgo: {risk_dist.get('Alto Riesgo', 0)} pacientes")
            st.write(f"- Riesgo Moderado: {risk_dist.get('Riesgo Moderado', 0)} pacientes")
            st.write(f"- Normal: {risk_dist.get('Normal', 0)} pacientes")
        
        st.markdown("### Ordenación")
        sorted_df, _ = processor.sort_data(['DEATH_EVENT', 'age'], ascending=False)
        st.dataframe(sorted_df.rename(columns=traducciones_dask), use_container_width=True)
    
    elif page == "📊 Visualizaciones":
        st.markdown("## Visualizaciones Python")
        
        visualizer = DataVisualizer(df)
        
        viz_type = st.selectbox("Tipo:", 
            ["Mortalidad", "Edad", "Correlación", "Comorbilidades"])
        
        if viz_type == "Mortalidad":
            st.markdown("### 🔍 Filtrar por Rango de Edad")
            age_range_viz = st.slider(
                "Selecciona rango de edad:", 
                40, 95, (40, 95),
                key="viz_age_range"
            )
            age_min_viz, age_max_viz = age_range_viz
            
            df_filtered = df[(df['age'] >= age_min_viz) & (df['age'] <= age_max_viz)]
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Pacientes en rango", len(df_filtered))
            col2.metric("Fallecidos", int(df_filtered['DEATH_EVENT'].sum()))
            
            tasa_mortalidad = (df_filtered['DEATH_EVENT'].mean() * 100) if len(df_filtered) > 0 else 0.0
            col3.metric("Tasa Mortalidad", f"{tasa_mortalidad:.1f}%")
            st.caption(f"Mostrando pacientes entre {age_min_viz} y {age_max_viz} años")
            
            visualizer_filtered = DataVisualizer(df_filtered)
            
            fig = visualizer_filtered.plot_target_distribution()
            st.pyplot(fig)
            # Dar margen de error si n es pequeña
            if len(df_filtered) < 20:
                st.warning("⚠️ Pocos pacientes en este rango. Los gráficos pueden no ser representativos.")
        elif viz_type == "Edad":
            fig = visualizer.plot_age_distribution()
            st.pyplot(fig)
        elif viz_type == "Correlación":
            fig = visualizer.plot_correlation_matrix()
            st.pyplot(fig)
        elif viz_type == "Comorbilidades":
            fig = visualizer.plot_comorbidities()
            st.pyplot(fig)

        st.markdown("---")
        st.markdown("### Visualizaciones con R")
        
        r_int = RPythonIntegration()
        
        if r_int.check_r_installed():
            st.success("Gráficos con R disponibles")
            
            if st.button("Generar gráficos con R"):
                with st.spinner("R está trabajando en los gráficos..."):
                    success = r_int.generate_r_visualizations('data/heart_failure_dataset.csv')
                    
                    if success:
                        ruta_grafico_r = Path("r_outputs/r_distributions.png")
                        if ruta_grafico_r.exists():
                            st.image(str(ruta_grafico_r), caption="Gráficos generados por R")
                        else:
                            st.warning("⚠️ El script de R terminó pero no encuentro la imagen.")
                    else:
                        st.error("❌ R falló al generar los gráficos. Revisa la terminal.")
        else:
            st.info("💡 R no está instalado. Instala R y añádelo al PATH.")
    elif page == "🤖 Modelos ML":
        st.markdown("## Modelización ML")
        
        if 'models' not in st.session_state:
            with st.spinner("Entrenando modelos..."):
                st.session_state.models = train_models(processor)
        
        models = st.session_state.models
        model_name = st.selectbox("Modelo:", 
            ['Random Forest', 'Logistic Regression', 'Gradient Boosting', 'SVM'])
        
        results = models.results[model_name]
        
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Accuracy", f"{results['accuracy']:.4f}")
        col2.metric("Precision", f"{results['precision']:.4f}")
        col3.metric("Recall", f"{results['recall']:.4f}")
        col4.metric("F1-Score", f"{results['f1']:.4f}")
        col5.metric("ROC-AUC", f"{results['roc_auc']:.4f}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Matriz de Confusión")
            fig, ax = plt.subplots()
            sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=ax)
            st.pyplot(fig)
        
        with col2:
            st.markdown("### Curva ROC")
            fig, ax = plt.subplots()
            roc = results['roc_curve']
            ax.plot(roc['fpr'], roc['tpr'], label=f'AUC={results["roc_auc"]:.4f}')
            ax.plot([0,1], [0,1], 'k--')
            ax.set_xlabel('FPR')
            ax.set_ylabel('TPR')
            ax.legend()
            st.pyplot(fig)
    elif page == "🧠 Deep Learning":
        st.markdown("## Deep Learning con Keras")
        
        if 'models' not in st.session_state:
            with st.spinner("Entrenando modelos..."):
                st.session_state.models = train_models(processor)
        
        models = st.session_state.models
        
        st.markdown("""
        **Arquitectura:** 3 capas (64, 32, 16) + Dropout + BatchNorm + L2
        
        **Optimizador:** Adam | **Loss:** Binary Crossentropy
        """)
        
        results = models.results['Deep Learning']
        
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Accuracy", f"{results['accuracy']:.4f}")
        col2.metric("Precision", f"{results['precision']:.4f}")
        col3.metric("Recall", f"{results['recall']:.4f}")
        col4.metric("F1-Score", f"{results['f1']:.4f}")
        col5.metric("ROC-AUC", f"{results['roc_auc']:.4f}")
        
        if 'history' in results:
            st.markdown("### Entrenamiento")
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots()
                ax.plot(results['history']['loss'], label='Train')
                ax.plot(results['history']['val_loss'], label='Val')
                ax.set_title('Loss')
                ax.legend()
                st.pyplot(fig)
            
            with col2:
                fig, ax = plt.subplots()
                ax.plot(results['history']['accuracy'], label='Train')
                ax.plot(results['history']['val_accuracy'], label='Val')
                ax.set_title('Accuracy')
                ax.legend()
                st.pyplot(fig)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Matriz de Confusión")
            fig, ax = plt.subplots()
            sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Greens', ax=ax)
            st.pyplot(fig)
        
        with col2:
            st.markdown("### Curva ROC")
            fig, ax = plt.subplots()
            roc = results['roc_curve']
            ax.plot(roc['fpr'], roc['tpr'], label=f'AUC={results["roc_auc"]:.4f}')
            ax.plot([0,1], [0,1], 'k--')
            ax.legend()
            st.pyplot(fig)
    elif page == "📉 Comparación":
        st.markdown("## 📉 Comparación de Modelos")
        
        if 'models' not in st.session_state:
            with st.spinner("Entrenando..."):
                st.session_state.models = train_models(processor)
        
        models = st.session_state.models
        comparison = models.compare_models()
        
        st.dataframe(comparison, use_container_width=True)
        
        st.markdown("### 🏆 Mejor Modelo por ROC-AUC")
        best = comparison.iloc[0]
        st.success(f"**{best['Model']}** con ROC-AUC de {best['ROC-AUC']:.4f}")
        
        st.markdown("---")
        
        # Boton de tiempos
        if st.button("⏱️ Comparar Tiempos de Entrenamiento", type="primary"):
            st.markdown("### ⚡ Velocidad de Entrenamiento")
            
            # Extraer tiempos
            tiempos = {}
            for model_name in models.results.keys():
                if 'training_time' in models.results[model_name]:
                    tiempos[model_name] = models.results[model_name]['training_time']
            
            # Ordenar por tiempo
            tiempos_sorted = dict(sorted(tiempos.items(), key=lambda x: x[1]))
            import plotly.graph_objects as go
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = go.Figure(data=[
                    go.Bar(
                        x=list(tiempos_sorted.keys()),
                        y=list(tiempos_sorted.values()),
                        text=[f"{t:.2f}s" for t in tiempos_sorted.values()],
                        textposition='outside',
                        marker_color=['#2ecc71' if i == 0 else '#3498db' for i in range(len(tiempos_sorted))]
                    )
                ])
                
                fig.update_layout(
                    title='Tiempo de Entrenamiento por Modelo',
                    xaxis_title='Modelo',
                    yaxis_title='Tiempo (segundos)',
                    height=400,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("**⏱️ Ranking de Velocidad:**")
                
                for idx, (model, tiempo) in enumerate(tiempos_sorted.items(), 1):
                    emoji = "🥇" if idx == 1 else "🥈" if idx == 2 else "🥉" if idx == 3 else f"{idx}️⃣"
                    st.metric(
                        f"{emoji} {model}",
                        f"{tiempo:.2f}s"
                    )
            
            fastest_model = list(tiempos_sorted.keys())[0]
            fastest_time = list(tiempos_sorted.values())[0]
            slowest_model = list(tiempos_sorted.keys())[-1]
            slowest_time = list(tiempos_sorted.values())[-1]
            
            st.success(f"🏆 **Modelo más rápido:** {fastest_model} ({fastest_time:.2f}s)")
            st.info(f"📊 **Diferencia:** {fastest_model} es {slowest_time/fastest_time:.1f}x más rápido que {slowest_model}")
    elif page == "🎯 Predictor":
        st.markdown("## Predictor Interactivo")
        
        if 'models' not in st.session_state:
            with st.spinner("Cargando modelos..."):
                st.session_state.models = train_models(processor)
        
        models = st.session_state.models
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Edad", 18, 120, 60)
            if age < 40 or age > 95:
                st.warning("⚠️ Edad fuera del rango de entrenamiento (40-95). Predicción menos confiable.")
            anaemia = st.selectbox("Anemia", [0, 1])
            cpk = st.number_input("CPK", 0, 8000, 250)
            diabetes = st.selectbox("Diabetes", [0, 1])
        
        with col2:
            ej_frac = st.slider("Fracción Eyección (%)", 0, 100, 40)
            if ej_frac < 14 or ej_frac > 80:
                st.warning("⚠️ Fracción eyección fuera del rango de entrenamiento (14-80).")
            high_bp = st.selectbox("Hipertensión", [0, 1])
            platelets = st.number_input("Plaquetas", 0, 900000, 250000)
            serum_creat = st.number_input("Creatinina", 0.0, 15.0, 1.0, 0.1)
            if serum_creat > 9.4:
                st.warning("⚠️ Creatinina fuera del rango de entrenamiento (0.5-9.4).")
        
        with col3:
            serum_sodium = st.number_input("Sodio", 90, 160, 137)
            if serum_sodium < 113 or serum_sodium > 148:
                st.warning("⚠️ Sodio fuera del rango de entrenamiento (113-148).")
            sex = st.selectbox("Sexo", [0, 1], format_func=lambda x: "Mujer" if x==0 else "Hombre")
            smoking = st.selectbox("Fumador", [0, 1])        
        model_choice = st.selectbox("Modelo:", 
            ['SVM', 'Logistic Regression', 'Random Forest', 'Gradient Boosting', 'Deep Learning'])
        
        if st.button("Riesgo cardíaco", type="primary"):
            patient = {
                'age': age, 'anaemia': anaemia, 'creatinine_phosphokinase': cpk,
                'diabetes': diabetes, 'ejection_fraction': ej_frac,
                'high_blood_pressure': high_bp, 'platelets': platelets,
                'serum_creatinine': serum_creat, 'serum_sodium': serum_sodium,
                'sex': sex, 'smoking': smoking
            }
            
            result = models.predict_single_patient(patient, model_choice)
            
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            
            pred_text = "Fallecimiento" if result['prediction'] == 1 else "Supervivencia"
            col1.metric("Predicción", pred_text)
            col2.metric("Probabilidad", f"{result['probability']*100:.1f}%")
            col3.metric("Nivel", result['risk_level'])
            
            risk = result['probability']
            st.progress(risk)
            
            if risk < 0.3:
                st.success("✅ Bajo riesgo. Indicadores favorables.")
            elif risk < 0.7:
                st.warning("⚠️ Riesgo moderado. Monitoreo recomendado.")
            else:
                st.error("🚨 Alto riesgo. Atención médica inmediata.")
    
    st.markdown("---")



if __name__ == '__main__':
    main()
