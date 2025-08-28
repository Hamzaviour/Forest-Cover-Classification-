"""
Forest Cover Classification - Professional Streamlit App
Created by Hamza Younas
"""

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
import os
from io import BytesIO
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Forest Cover Classification",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 3rem 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        position: relative;
        overflow: hidden;
    }
    
    .main-header-card::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: repeating-linear-gradient(
            0deg,
            transparent,
            transparent 2px,
            rgba(255,255,255,0.03) 2px,
            rgba(255,255,255,0.03) 4px
        );
        animation: shimmer 20s linear infinite;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-50%) translateY(-50%) rotate(0deg); }
        100% { transform: translateX(-50%) translateY(-50%) rotate(360deg); }
    }
    
    .header-content {
        position: relative;
        z-index: 2;
    }
    
    .main-title {
        font-size: 3.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .subtitle {
        font-size: 1.3rem;
        font-weight: 400;
        margin-bottom: 0.5rem;
        opacity: 0.9;
    }
    
    .company-tag {
        background: rgba(255,255,255,0.2);
        padding: 0.5rem 1.5rem;
        border-radius: 25px;
        font-size: 1rem;
        font-weight: 600;
        display: inline-block;
        margin-top: 1rem;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.3);
    }
    
    .forest-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
        display: block;
    }
    
    .section-header {
        font-size: 2rem;
        font-weight: bold;
        color: #1e3c72;
        border-bottom: 3px solid #2a5298;
        padding-bottom: 0.5rem;
        margin: 2rem 0 1rem 0;
    }
    
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255,255,255,0.2);
        margin: 1rem 0;
        backdrop-filter: blur(10px);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.15);
        transform: translateY(0);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 25px rgba(0, 0, 0, 0.2);
    }
    
    .feature-importance {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .creator-badge {
        position: fixed;
        bottom: 20px;
        right: 20px;
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        padding: 0.8rem 1.5rem;
        border-radius: 25px;
        font-size: 0.9rem;
        font-weight: bold;
        z-index: 1000;
        box-shadow: 0 6px 15px rgba(0,0,0,0.2);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .stSelectbox > div > div {
        background-color: #f8f9fa;
        border-radius: 10px;
    }
    
    /* Enhanced sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Navigation styling */
    .nav-item {
        padding: 0.5rem 1rem;
        margin: 0.2rem 0;
        border-radius: 10px;
        transition: background 0.3s ease;
    }
    
    .nav-item:hover {
        background: rgba(102, 126, 234, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Creator badge
st.markdown("""
<div class="creator-badge">
    Created by Hamza Younas
</div>
""", unsafe_allow_html=True)

# Load data and models
@st.cache_data
def load_data():
    """Load and preprocess the forest cover dataset"""
    try:
        column_names = [
            'Elevation','Aspect','Slope','Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology',
            'Horizontal_Distance_To_Roadways','Hillshade_9am','Hillshade_Noon','Hillshade_3pm',
            'Horizontal_Distance_To_Fire_Points','Wilderness_Area1','Wilderness_Area2','Wilderness_Area3',
            'Wilderness_Area4','Soil_Type1','Soil_Type2','Soil_Type3','Soil_Type4','Soil_Type5','Soil_Type6',
            'Soil_Type7','Soil_Type8','Soil_Type9','Soil_Type10','Soil_Type11','Soil_Type12','Soil_Type13',
            'Soil_Type14','Soil_Type15','Soil_Type16','Soil_Type17','Soil_Type18','Soil_Type19','Soil_Type20',
            'Soil_Type21','Soil_Type22','Soil_Type23','Soil_Type24','Soil_Type25','Soil_Type26','Soil_Type27',
            'Soil_Type28','Soil_Type29','Soil_Type30','Soil_Type31','Soil_Type32','Soil_Type33','Soil_Type34',
            'Soil_Type35','Soil_Type36','Soil_Type37','Soil_Type38','Soil_Type39','Soil_Type40','Cover_Type'
        ]
        
        df = pd.read_csv('covtype.csv', header=None, names=column_names)
        df['Cover_Type_numeric'] = pd.to_numeric(df['Cover_Type'], errors='coerce')
        df_clean = df[df['Cover_Type_numeric'].notna()].copy()
        
        return df_clean
    except FileNotFoundError:
        st.error("❌ Dataset file 'covtype.csv' not found. Please ensure the file is in the same directory.")
        return None

@st.cache_resource
def load_models_and_train():
    """Load or train models"""
    models = {}
    metrics = {}
    
    # Try to load existing models
    try:
        if os.path.exists('models/random_forest_model.pkl'):
            with open('models/random_forest_model.pkl', 'rb') as f:
                models['Random Forest'] = pickle.load(f)
            with open('models/xgboost_model.pkl', 'rb') as f:
                models['XGBoost'] = pickle.load(f)
            with open('models/decision_tree_model.pkl', 'rb') as f:
                models['Decision Tree'] = pickle.load(f)
            with open('models/model_metrics.pkl', 'rb') as f:
                metrics = pickle.load(f)
            return models, metrics
    except:
        pass
    
    # Train models if not found
    df = load_data()
    if df is not None:
        X = df.drop(['Cover_Type', 'Cover_Type_numeric'], axis=1)
        y = df['Cover_Type_numeric'].astype(int) - 1
        
        X = X.apply(pd.to_numeric, errors='coerce')
        X = X.fillna(X.median())
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Train Random Forest
        rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_clf.fit(X_train, y_train)
        y_pred_rf = rf_clf.predict(X_test)
        
        # Train XGBoost
        xgb_clf = XGBClassifier(objective='multi:softmax', num_class=int(y.nunique()), 
                               use_label_encoder=False, eval_metric='mlogloss', random_state=42, n_jobs=-1, verbosity=0)
        xgb_clf.fit(X_train, y_train)
        y_pred_xgb = xgb_clf.predict(X_test)
        
        # Train Decision Tree
        dt_clf = DecisionTreeClassifier(random_state=42, max_depth=20, min_samples_split=10, min_samples_leaf=5)
        dt_clf.fit(X_train, y_train)
        y_pred_dt = dt_clf.predict(X_test)
        
        models = {
            'Random Forest': rf_clf,
            'XGBoost': xgb_clf,
            'Decision Tree': dt_clf
        }
        
        metrics = {
            'Random Forest': {
                'accuracy': accuracy_score(y_test, y_pred_rf),
                'confusion_matrix': confusion_matrix(y_test, y_pred_rf),
                'feature_importance': dict(zip(X.columns, rf_clf.feature_importances_))
            },
            'XGBoost': {
                'accuracy': accuracy_score(y_test, y_pred_xgb),
                'confusion_matrix': confusion_matrix(y_test, y_pred_xgb),
                'feature_importance': dict(zip(X.columns, xgb_clf.feature_importances_))
            },
            'Decision Tree': {
                'accuracy': accuracy_score(y_test, y_pred_dt),
                'confusion_matrix': confusion_matrix(y_test, y_pred_dt),
                'feature_importance': dict(zip(X.columns, dt_clf.feature_importances_))
            }
        }
        
        return models, metrics
    
    return {}, {}

# Main App Header with Professional Card Design
st.markdown("""
<div class="main-header-card">
    <div class="header-content">
        <div class="forest-icon">🌲🌿🍃</div>
        <h1 class="main-title">Forest Cover Classification</h1>
        <p class="subtitle">Advanced Machine Learning Analysis using Random Forest, XGBoost, and Decision Tree algorithms</p>
        <div class="company-tag">Elevvo Pathways</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🎯 Navigation")
section = st.sidebar.selectbox(
    "Choose a section:",
    ["📊 Data Exploration", "📈 Model Comparison", "🎯 Performance Analysis", "🔮 Predictions"]
)

# Load data and models
df = load_data()
models, metrics = load_models_and_train()

if df is None:
    st.stop()

# Feature definitions for better understanding
cover_type_names = {
    1: "Spruce/Fir",
    2: "Lodgepole Pine", 
    3: "Ponderosa Pine",
    4: "Cottonwood/Willow",
    5: "Aspen",
    6: "Douglas-fir",
    7: "Krummholz"
}

# Section 1: Data Exploration
if section == "📊 Data Exploration":
    st.markdown('<h2 class="section-header">📊 Data Exploration</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Total Samples</h3>
            <h2>{:,}</h2>
        </div>
        """.format(len(df)), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🔢 Features</h3>
            <h2>{}</h2>
        </div>
        """.format(df.shape[1] - 2), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🌲 Cover Types</h3>
            <h2>7</h2>
        </div>
        """.format(), unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>💾 Dataset Size</h3>
            <h2>{:.1f} MB</h2>
        </div>
        """.format(df.memory_usage(deep=True).sum() / 1024 / 1024), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Dataset Overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📋 Dataset Overview")
        st.dataframe(df.head(10), use_container_width=True)
    
    with col2:
        st.markdown("### 📈 Cover Type Distribution")
        cover_counts = df['Cover_Type_numeric'].value_counts().sort_index()
        
        fig = px.bar(
            x=[cover_type_names[i] for i in cover_counts.index], 
            y=cover_counts.values,
            color=cover_counts.values,
            color_continuous_scale="viridis",
            title="Distribution of Forest Cover Types"
        )
        fig.update_layout(
            xaxis_title="Cover Type",
            yaxis_title="Count",
            showlegend=False,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature Statistics
    st.markdown("### 📊 Feature Statistics")
    
    # Select continuous features for analysis
    continuous_features = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 
                          'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways']
    
    selected_feature = st.selectbox("Select feature to analyze:", continuous_features)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Convert to numeric if needed
        feature_data = pd.to_numeric(df[selected_feature], errors='coerce').dropna()
        
        fig = px.histogram(
            x=feature_data,
            nbins=50,
            title=f"Distribution of {selected_feature}",
            color_discrete_sequence=["#667eea"]
        )
        fig.update_layout(
            xaxis_title=selected_feature,
            yaxis_title="Frequency",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Box plot by cover type
        plot_df = df[[selected_feature, 'Cover_Type_numeric']].copy()
        plot_df[selected_feature] = pd.to_numeric(plot_df[selected_feature], errors='coerce')
        plot_df = plot_df.dropna()
        plot_df['Cover_Type_Name'] = plot_df['Cover_Type_numeric'].map(cover_type_names)
        
        fig = px.box(
            plot_df, 
            x='Cover_Type_Name', 
            y=selected_feature,
            title=f"{selected_feature} by Cover Type",
            color='Cover_Type_Name'
        )
        fig.update_layout(
            xaxis_title="Cover Type",
            yaxis_title=selected_feature,
            showlegend=False,
            height=400
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

# Section 2: Model Comparison
elif section == "📈 Model Comparison":
    st.markdown('<h2 class="section-header">📈 Model Comparison</h2>', unsafe_allow_html=True)
    
    if metrics:
        # Performance metrics
        col1, col2, col3 = st.columns(3)
        
        accuracies = {name: metric['accuracy'] for name, metric in metrics.items()}
        
        with col1:
            rf_acc = accuracies.get('Random Forest', 0)
            st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
                <h3>🌳 Random Forest</h3>
                <h2>{rf_acc:.3f}</h2>
                <p>{rf_acc*100:.1f}% Accuracy</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            xgb_acc = accuracies.get('XGBoost', 0)
            st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
                <h3>🚀 XGBoost</h3>
                <h2>{xgb_acc:.3f}</h2>
                <p>{xgb_acc*100:.1f}% Accuracy</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            dt_acc = accuracies.get('Decision Tree', 0)
            st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
                <h3>🌲 Decision Tree</h3>
                <h2>{dt_acc:.3f}</h2>
                <p>{dt_acc*100:.1f}% Accuracy</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Comparison Chart
        st.markdown("### 📊 Performance Comparison")
        
        model_names = list(accuracies.keys())
        accuracy_values = list(accuracies.values())
        
        fig = go.Figure(data=[
            go.Bar(
                x=model_names,
                y=accuracy_values,
                marker_color=['#667eea', '#f5576c', '#4facfe'],
                text=[f'{acc:.3f}<br>({acc*100:.1f}%)' for acc in accuracy_values],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="Model Accuracy Comparison",
            xaxis_title="Models",
            yaxis_title="Accuracy Score",
            yaxis_range=[0.8, 1.0],
            height=500,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Best Model Highlight
        best_model = max(accuracies, key=accuracies.get)
        best_accuracy = accuracies[best_model]
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    color: white; padding: 1rem; border-radius: 10px; text-align: center; margin: 1rem 0;">
            <h3>🏆 Best Performing Model</h3>
            <h2>{best_model}</h2>
            <p>Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Model Analysis
        st.markdown("### 🔍 Model Analysis")
        
        analysis_text = """
        **Random Forest** excels with ensemble learning, combining multiple decision trees to reduce overfitting and improve generalization.
        
        **XGBoost** uses gradient boosting for efficient learning, though it may require more hyperparameter tuning for optimal performance.
        
        **Decision Tree** provides high interpretability but may be prone to overfitting on complex datasets like this one.
        
        The Random Forest model shows superior performance due to its robustness against overfitting and ability to handle complex feature interactions.
        """
        
        st.markdown(analysis_text)
    
    else:
        st.error("❌ Models not trained yet. Please check if the dataset is available.")

# Section 3: Performance Analysis
elif section == "🎯 Performance Analysis":
    st.markdown('<h2 class="section-header">🎯 Performance Analysis</h2>', unsafe_allow_html=True)
    
    if metrics:
        # Model selector
        selected_model = st.selectbox("Select Model for Detailed Analysis:", list(metrics.keys()))
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"### 🎯 Confusion Matrix - {selected_model}")
            
            # Confusion Matrix
            cm = metrics[selected_model]['confusion_matrix']
            
            fig = px.imshow(
                cm,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="Blues",
                title=f"Confusion Matrix - {selected_model}"
            )
            
            fig.update_layout(
                xaxis_title="Predicted",
                yaxis_title="Actual",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown(f"### 📊 Feature Importance - {selected_model}")
            
            # Feature Importance
            importance_data = metrics[selected_model]['feature_importance']
            importance_df = pd.DataFrame(
                list(importance_data.items()), 
                columns=['Feature', 'Importance']
            ).sort_values('Importance', ascending=False).head(15)
            
            fig = px.bar(
                importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title=f"Top 15 Features - {selected_model}",
                color='Importance',
                color_continuous_scale="viridis"
            )
            
            fig.update_layout(
                height=500,
                yaxis={'categoryorder': 'total ascending'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed Metrics
        st.markdown("### 📈 Detailed Performance Metrics")
        
        accuracy = metrics[selected_model]['accuracy']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🎯 Accuracy", f"{accuracy:.4f}", f"{accuracy*100:.2f}%")
        
        with col2:
            # Calculate precision (weighted average)
            cm = metrics[selected_model]['confusion_matrix']
            precision = np.diagonal(cm) / np.sum(cm, axis=0)
            weighted_precision = np.average(precision, weights=np.sum(cm, axis=1))
            st.metric("🔍 Precision", f"{weighted_precision:.4f}", f"{weighted_precision*100:.2f}%")
        
        with col3:
            # Calculate recall (weighted average)
            recall = np.diagonal(cm) / np.sum(cm, axis=1)
            weighted_recall = np.average(recall, weights=np.sum(cm, axis=1))
            st.metric("📡 Recall", f"{weighted_recall:.4f}", f"{weighted_recall*100:.2f}%")
    
    else:
        st.error("❌ Performance metrics not available. Please check if models are trained.")

# Section 4: Predictions
elif section == "🔮 Predictions":
    st.markdown('<h2 class="section-header">🔮 Interactive Predictions</h2>', unsafe_allow_html=True)
    
    if models:
        # Model selector for predictions
        prediction_model = st.selectbox("Select Model for Prediction:", list(models.keys()))
        
        st.markdown("### 🎛️ Input Features")
        st.markdown("Adjust the sliders below to set feature values for prediction:")
        
        # Create input form
        col1, col2, col3 = st.columns(3)
        
        with col1:
            elevation = st.slider("🏔️ Elevation", 1859, 3858, 2800)
            aspect = st.slider("🧭 Aspect", 0, 360, 180)
            slope = st.slider("📐 Slope", 0, 52, 15)
            
        with col2:
            h_dist_hydrology = st.slider("💧 Distance to Hydrology", 0, 1343, 200)
            v_dist_hydrology = st.slider("⬆️ Vertical Distance to Hydrology", -173, 554, 0)
            h_dist_roadways = st.slider("🛣️ Distance to Roadways", 0, 6890, 1000)
            
        with col3:
            hillshade_9am = st.slider("🌅 Hillshade 9am", 0, 254, 200)
            hillshade_noon = st.slider("☀️ Hillshade Noon", 99, 254, 230)
            hillshade_3pm = st.slider("🌇 Hillshade 3pm", 0, 248, 135)
        
        h_dist_fire = st.slider("🔥 Distance to Fire Points", 0, 6993, 1000)
        
        # Wilderness areas (simplified - user can select one)
        wilderness_area = st.selectbox("🏞️ Wilderness Area", 
                                     ["Rawah", "Neota", "Comanche Peak", "Cache la Poudre"])
        
        # Soil type (simplified selection)
        soil_type = st.selectbox("🌱 Soil Type", 
                                [f"Soil Type {i}" for i in range(1, 41)])
        
        # Predict button
        if st.button("🔮 Make Prediction", type="primary"):
            # Prepare input data
            input_data = np.zeros(54)  # 54 features total
            
            # Set continuous features
            input_data[0] = elevation
            input_data[1] = aspect
            input_data[2] = slope
            input_data[3] = h_dist_hydrology
            input_data[4] = v_dist_hydrology
            input_data[5] = h_dist_roadways
            input_data[6] = hillshade_9am
            input_data[7] = hillshade_noon
            input_data[8] = hillshade_3pm
            input_data[9] = h_dist_fire
            
            # Set wilderness area (one-hot encoded)
            wilderness_mapping = {"Rawah": 10, "Neota": 11, "Comanche Peak": 12, "Cache la Poudre": 13}
            if wilderness_area in wilderness_mapping:
                input_data[wilderness_mapping[wilderness_area]] = 1
            
            # Set soil type (one-hot encoded)
            soil_index = int(soil_type.split()[-1]) - 1 + 14  # Soil types start at index 14
            if 14 <= soil_index < 54:
                input_data[soil_index] = 1
            
            # Make prediction
            model = models[prediction_model]
            prediction = model.predict([input_data])[0]
            prediction_proba = model.predict_proba([input_data])[0] if hasattr(model, 'predict_proba') else None
            
            # Display result
            predicted_cover_type = cover_type_names[prediction + 1]
            
            st.markdown("---")
            st.markdown("### 🎯 Prediction Result")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            color: white; padding: 2rem; border-radius: 15px; text-align: center;">
                    <h3>🌲 Predicted Forest Cover Type</h3>
                    <h1>{predicted_cover_type}</h1>
                    <p>Class {prediction + 1}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                if prediction_proba is not None:
                    st.markdown("### 📊 Prediction Confidence")
                    
                    # Create probability chart
                    prob_df = pd.DataFrame({
                        'Cover Type': [cover_type_names[i+1] for i in range(7)],
                        'Probability': prediction_proba
                    })
                    
                    fig = px.bar(
                        prob_df,
                        x='Cover Type',
                        y='Probability',
                        color='Probability',
                        color_continuous_scale="viridis",
                        title="Prediction Probabilities"
                    )
                    
                    fig.update_layout(
                        xaxis_title="Cover Type",
                        yaxis_title="Probability",
                        showlegend=False,
                        height=400
                    )
                    fig.update_xaxes(tickangle=45)
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        # Sample predictions
        st.markdown("---")
        st.markdown("### 📋 Quick Sample Predictions")
        
        sample_button = st.button("🎲 Generate Random Sample Prediction")
        
        if sample_button and df is not None:
            # Get a random sample from the dataset
            sample_idx = np.random.randint(0, len(df))
            sample_row = df.iloc[sample_idx]
            
            # Prepare sample data
            X_sample = df.drop(['Cover_Type', 'Cover_Type_numeric'], axis=1).iloc[sample_idx:sample_idx+1]
            X_sample_numeric = X_sample.apply(pd.to_numeric, errors='coerce').fillna(0)
            
            actual_type = int(sample_row['Cover_Type_numeric'])
            
            # Make predictions with all models
            predictions = {}
            for model_name, model in models.items():
                pred = model.predict(X_sample_numeric)[0]
                predictions[model_name] = cover_type_names[pred + 1]
            
            st.markdown(f"**Actual Cover Type:** {cover_type_names[actual_type]}")
            st.markdown("**Model Predictions:**")
            
            for model_name, pred in predictions.items():
                correct = "✅" if pred == cover_type_names[actual_type] else "❌"
                st.markdown(f"- {model_name}: {pred} {correct}")
    
    else:
        st.error("❌ Models not available for predictions. Please check if models are trained.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <p>🌲 Forest Cover Classification Dashboard | Built with Streamlit</p>
    <p>Machine Learning Models: Random Forest, XGBoost, Decision Tree</p>
    <p><strong>Created by Hamza Younas | Elevvo Pathways</strong></p>
</div>
""", unsafe_allow_html=True)
