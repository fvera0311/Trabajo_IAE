"""
Módulo de modelización con ML tradicional y Deep Learning
Cubre el requisito 7: Modelización + Elemento avanzado: Deep Learning
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
from imblearn.over_sampling import SMOTE
import joblib
from pathlib import Path

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks


class HeartFailureModels:
    """Clase para entrenar y evaluar modelos de predicción"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.results = {}
        
    def prepare_data(self, X, y, test_size=0.2, use_smote=True):
        """Prepara datos para entrenamiento"""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        if use_smote:
            smote = SMOTE(random_state=self.random_state)
            self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)
            print(f"✅ SMOTE aplicado - Muestras: {len(self.X_train)}")
        
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"✅ Datos preparados: Train={self.X_train.shape[0]}, Test={self.X_test.shape[0]}")
        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test
    
    def train_random_forest(self, n_estimators=100, max_depth=10):
        """Entrena Random Forest"""
        print("\n🌲 Entrenando Random Forest...")
        
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        rf.fit(self.X_train_scaled, self.y_train)
        self.models['Random Forest'] = rf
        
        y_pred = rf.predict(self.X_test_scaled)
        y_pred_proba = rf.predict_proba(self.X_test_scaled)[:, 1]
        
        self.results['Random Forest'] = self._evaluate_model(y_pred, y_pred_proba, 'Random Forest')
        self.results['Random Forest']['feature_importance'] = rf.feature_importances_
        
        print(f"✅ Random Forest - Accuracy: {self.results['Random Forest']['accuracy']:.4f}")
        return rf
    
    def train_logistic_regression(self, C=1.0):
        """Entrena Logistic Regression"""
        print("\n📈 Entrenando Logistic Regression...")
        
        lr = LogisticRegression(C=C, random_state=self.random_state, max_iter=1000)
        lr.fit(self.X_train_scaled, self.y_train)
        self.models['Logistic Regression'] = lr
        
        y_pred = lr.predict(self.X_test_scaled)
        y_pred_proba = lr.predict_proba(self.X_test_scaled)[:, 1]
        
        self.results['Logistic Regression'] = self._evaluate_model(y_pred, y_pred_proba, 'Logistic Regression')
        
        print(f"✅ Logistic Regression - Accuracy: {self.results['Logistic Regression']['accuracy']:.4f}")
        return lr
    
    def train_gradient_boosting(self, n_estimators=100, learning_rate=0.1):
        """Entrena Gradient Boosting"""
        print("\n🚀 Entrenando Gradient Boosting...")
        
        gb = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=self.random_state
        )
        
        gb.fit(self.X_train_scaled, self.y_train)
        self.models['Gradient Boosting'] = gb
        
        y_pred = gb.predict(self.X_test_scaled)
        y_pred_proba = gb.predict_proba(self.X_test_scaled)[:, 1]
        
        self.results['Gradient Boosting'] = self._evaluate_model(y_pred, y_pred_proba, 'Gradient Boosting')
        
        print(f"✅ Gradient Boosting - Accuracy: {self.results['Gradient Boosting']['accuracy']:.4f}")
        return gb
    
    def train_svm(self, C=1.0, kernel='rbf'):
        """Entrena Support Vector Machine"""
        print("\n🎯 Entrenando SVM...")
        
        svm = SVC(C=C, kernel=kernel, probability=True, random_state=self.random_state)
        svm.fit(self.X_train_scaled, self.y_train)
        self.models['SVM'] = svm
        
        y_pred = svm.predict(self.X_test_scaled)
        y_pred_proba = svm.predict_proba(self.X_test_scaled)[:, 1]
        
        self.results['SVM'] = self._evaluate_model(y_pred, y_pred_proba, 'SVM')
        
        print(f"✅ SVM - Accuracy: {self.results['SVM']['accuracy']:.4f}")
        return svm
    
    # ============ DEEP LEARNING ============
    
    def build_neural_network(self, input_dim, layers_config=[64, 32, 16]):
        """Construye red neuronal para Deep Learning"""
        model = keras.Sequential()
        
        # Primera capa
        model.add(layers.Dense(
            layers_config[0], 
            activation='relu', 
            input_dim=input_dim,
            kernel_regularizer=keras.regularizers.l2(0.001)
        ))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.3))
        
        # Capas ocultas
        for units in layers_config[1:]:
            model.add(layers.Dense(units, activation='relu',
                                  kernel_regularizer=keras.regularizers.l2(0.001)))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(0.3))
        
        # Capa de salida
        model.add(layers.Dense(1, activation='sigmoid'))
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )
        
        return model
    
    def train_deep_learning(self, epochs=100, batch_size=16, validation_split=0.2):
        """Entrena modelo de Deep Learning"""
        print("\n🧠 Entrenando Red Neuronal (Deep Learning)...")
        
        input_dim = self.X_train_scaled.shape[1]
        model = self.build_neural_network(input_dim)
        
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss', patience=15, restore_best_weights=True, verbose=0
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=0
        )
        
        history = model.fit(
            self.X_train_scaled, self.y_train,
            epochs=epochs, batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )
        
        self.models['Deep Learning'] = model
        
        y_pred_proba = model.predict(self.X_test_scaled, verbose=0).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        self.results['Deep Learning'] = self._evaluate_model(y_pred, y_pred_proba, 'Deep Learning')
        self.results['Deep Learning']['history'] = history.history
        
        print(f"✅ Deep Learning - Accuracy: {self.results['Deep Learning']['accuracy']:.4f}")
        return model, history
    
    def _evaluate_model(self, y_pred, y_pred_proba, model_name):
        """Evalúa modelo y retorna métricas"""
        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred, zero_division=0),
            'recall': recall_score(self.y_test, y_pred, zero_division=0),
            'f1': f1_score(self.y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(self.y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(self.y_test, y_pred),
            'classification_report': classification_report(self.y_test, y_pred, zero_division=0),
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        fpr, tpr, thresholds = roc_curve(self.y_test, y_pred_proba)
        metrics['roc_curve'] = {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds}
        
        return metrics
    
    def compare_models(self):
        """Compara todos los modelos"""
        comparison = pd.DataFrame({
            'Model': list(self.results.keys()),
            'Accuracy': [self.results[m]['accuracy'] for m in self.results.keys()],
            'Precision': [self.results[m]['precision'] for m in self.results.keys()],
            'Recall': [self.results[m]['recall'] for m in self.results.keys()],
            'F1-Score': [self.results[m]['f1'] for m in self.results.keys()],
            'ROC-AUC': [self.results[m]['roc_auc'] for m in self.results.keys()]
        })
        
        comparison = comparison.sort_values('ROC-AUC', ascending=False).reset_index(drop=True)
        return comparison
    
    def predict_single_patient(self, patient_data, model_name='Random Forest'):
        """Predice para un paciente individual"""
        if isinstance(patient_data, dict):
            patient_data = pd.DataFrame([patient_data])
        
        patient_scaled = self.scaler.transform(patient_data)
        model = self.models[model_name]
        
        if model_name == 'Deep Learning':
            proba = model.predict(patient_scaled, verbose=0).flatten()[0]
            prediction = int(proba > 0.5)
        else:
            prediction = model.predict(patient_scaled)[0]
            proba = model.predict_proba(patient_scaled)[0, 1]
        
        return {
            'prediction': prediction,
            'probability': proba,
            'risk_level': 'Alto Riesgo' if proba > 0.7 else 'Riesgo Moderado' if proba > 0.4 else 'Bajo Riesgo'
        }
