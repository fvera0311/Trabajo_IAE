"""
Aplicación Streamlit - Predicción de Mortalidad por Insuficiencia Cardíaca
Requisito 8: Comunicación
"""
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
        ["🏠 Inicio", "🔬 Procesamiento Dask", "📊 Visualizaciones", 
         "🤖 Modelos ML", "🧠 Deep Learning", "📉 Comparación", "🎯 Predictor"]
    )
    
    processor, df = load_data()
    
    if page == "🏠 Inicio":
        st.markdown("## Información del Dataset")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Pacientes", len(df))
        col2.metric("Variables", len(df.columns))
        col3.metric("Fallecidos", df['DEATH_EVENT'].sum())
        col4.metric("Tasa Mortalidad", f"{(df['DEATH_EVENT'].mean()*100):.1f}%")
        
        st.markdown("### 📊 Dataset")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.markdown("### 📈 Estadísticas")
        st.dataframe(df.describe(), use_container_width=True)
    
    elif page == "🔬 Procesamiento Dask":
        st.markdown("## Procesamiento con Dask")
        
        st.markdown("### Filtro de edad")
        
        age_range = st.slider("Rango de edad:", 40, 95, (40, 95))
        age_min, age_max = age_range
        
        diabetes_filter = st.checkbox("Solo diabéticos")
        conditions = {'age': lambda x: (x >= age_min) & (x <= age_max)}
        if diabetes_filter:
            conditions['diabetes'] = lambda x: x == 1
        
        filtered = processor.filter_data(conditions)
        st.write(f"**Filtrados:** {len(filtered)} registros")
        st.dataframe(filtered, use_container_width=True)
        
        st.markdown("### Agregación")
        aggs = processor.aggregate_data()
        st.dataframe(aggs['por_muerte'], use_container_width=True)
        
        st.markdown("### Mapeo")
        mapped = processor.map_transformations()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📌 Transformación 1: Categorías de Edad**")
            st.dataframe(mapped[['age', 'age_category']].head(10))
            
            # Distribución de categorías
            age_dist = mapped['age_category'].value_counts()
            st.write("**Distribución:**")
            st.write(f"- Anciano: {age_dist.get('Anciano', 0)} pacientes")
            st.write(f"- Mediana Edad: {age_dist.get('Mediana Edad', 0)} pacientes")
            st.write(f"- Joven: {age_dist.get('Joven', 0)} pacientes")
        
        with col2:
            st.write("**📌 Transformación 2: Riesgo por Eyección**")
            st.dataframe(mapped[['ejection_fraction', 'ejection_risk']].head(10))
            
            # Distribución de riesgo
            risk_dist = mapped['ejection_risk'].value_counts()
            st.write("**Distribución:**")
            st.write(f"- Alto Riesgo: {risk_dist.get('Alto Riesgo', 0)} pacientes")
            st.write(f"- Riesgo Moderado: {risk_dist.get('Riesgo Moderado', 0)} pacientes")
            st.write(f"- Normal: {risk_dist.get('Normal', 0)} pacientes")
        
        st.markdown("### Ordenación")
        sorted_df, _ = processor.sort_data(['DEATH_EVENT', 'age'], ascending=False)
        st.dataframe(sorted_df, use_container_width=True)
    
    elif page == "📊 Visualizaciones":
        st.markdown("## Visualizaciones Python")
        
        visualizer = DataVisualizer(df)
        
        viz_type = st.selectbox("Tipo:", 
            ["Mortalidad", "Edad", "Correlación", "Comorbilidades"])
        
        if viz_type == "Mortalidad":
            # 1. Filtro por edad
            st.markdown("### 🔍 Filtrar por Rango de Edad")
            age_range_viz = st.slider(
                "Selecciona rango de edad:", 
                40, 95, (40, 95),
                key="viz_age_range"
            )
            age_min_viz, age_max_viz = age_range_viz
            st.markdown("### 🩺 Perfil")
            c1, c2, c3, c4, c5 = st.columns(5)
            filtro_diabetes = c1.checkbox("Diabético", value=False, key="chk_diab")
            filtro_anemia = c2.checkbox("Anemia", value=False, key="chk_anem")
            filtro_hipertension = c3.checkbox("Hipertensión", value=False, key="chk_hiper")
            filtro_fumador = c4.checkbox("Fumador", value=False, key="chk_fum")
            filtro_fallecido = c5.checkbox("Fallecido", value=False, key="chk_fall")
            df_filtered = df[
                (df['age'] >= age_min_viz) & (df['age'] <= age_max_viz) &
                (df['diabetes'] == int(filtro_diabetes)) &
                (df['anaemia'] == int(filtro_anemia)) &
                (df['high_blood_pressure'] == int(filtro_hipertension)) &
                (df['smoking'] == int(filtro_fumador)) &
                (df['DEATH_EVENT'] == int(filtro_fallecido))
            ]
            col1, col2, col3 = st.columns(3)
            col1.metric("Pacientes en rango", len(df_filtered))
            col2.metric("Fallecidos", df_filtered['DEATH_EVENT'].sum())
            col3.metric("Tasa Mortalidad", f"{(df_filtered['DEATH_EVENT'].mean()*100):.1f}%")
            visualizer_filtered = DataVisualizer(df_filtered)
            fig = visualizer_filtered.plot_target_distribution()
            st.pyplot(fig)
            # Mensaje informativo
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
            st.success("✅ ¡R detectado en el sistema!")
            
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
        st.markdown("## Comparación de Modelos")
        
        if 'models' not in st.session_state:
            with st.spinner("Entrenando..."):
                st.session_state.models = train_models(processor)
        
        models = st.session_state.models
        comparison = models.compare_models()
        
        st.dataframe(comparison, use_container_width=True)
        
        st.markdown("### 🏆 Mejor Modelo")
        best = comparison.iloc[0]
        st.success(f"**{best['Model']}** con ROC-AUC de {best['ROC-AUC']:.4f}")
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
            
            pred_text = "Alto Riesgo ⚠️" if result['prediction'] == 1 else "Bajo Riesgo ✅"
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
    st.markdown("❤️ **Universidad de Sevilla** · Curso 2025-26")


if __name__ == '__main__':
    main()
