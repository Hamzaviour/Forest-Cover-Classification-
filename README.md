# 🌲 Forest Cover Classification System

A professional Streamlit application for forest cover type classification using advanced machine learning algorithms.

**Created by Hamza Younas | Elevvo Pathways**

![Forest Cover Classification](https://img.shields.io/badge/ML-Forest%20Classification-green)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)

## 🚀 Live Demo

🌐 **[Try the Live Application](https://forest-cover-classification-by-hamza.streamlit.app/)**

## Features

### 📊 Data Exploration
- Interactive dataset overview and statistics
- Feature distribution visualizations
- Cover type analysis with beautiful charts

### 📈 Model Comparison  
- Performance comparison of Random Forest, XGBoost, and Decision Tree
- Accuracy metrics and visual comparisons
- Best model identification

### 🎯 Performance Analysis
- Detailed confusion matrices for each model
- Feature importance rankings
- Comprehensive performance metrics

### 🔮 Predictions
- Interactive prediction interface
- Real-time model predictions
- Sample data testing capabilities

## Setup Instructions

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Dataset**
   - Ensure `covtype.csv` is in the same directory as `app.py`
   - The dataset should contain forest cover type data

3. **Run the Application**
   ```bash
   streamlit run app.py
   ```

4. **Optional: Train Models First**
   ```bash
   python train_models.py
   ```
   This will save trained models to the `models/` directory for faster loading.

## Models Used

- **Random Forest Classifier**: Ensemble method with high accuracy
- **XGBoost Classifier**: Gradient boosting for efficient learning  
- **Decision Tree Classifier**: Interpretable tree-based model

## Technologies

- **Streamlit**: Web application framework
- **Plotly**: Interactive visualizations
- **Scikit-learn**: Machine learning algorithms
- **XGBoost**: Gradient boosting framework
- **Pandas/NumPy**: Data manipulation
- **Seaborn**: Statistical visualizations

## Application Structure

```
├── app.py              # Main Streamlit application
├── train_models.py     # Model training script
├── requirements.txt    # Python dependencies
├── README.md          # This file
├── covtype.csv        # Dataset (user provided)
└── models/            # Saved models directory (auto-created)
```

## Usage

1. Navigate through sections using the sidebar
2. Explore data patterns in the Data Exploration section
3. Compare model performance in Model Comparison
4. Analyze detailed metrics in Performance Analysis
5. Make predictions using the interactive Predictions interface



## 📊 Screenshots

*Add screenshots of your application here*



## 🙏 Acknowledgments

- **Elevvo Pathways** for project support
- UCI Machine Learning Repository for the Forest Cover Type dataset
- Streamlit team for the amazing framework

## 📧 Contact

**Hamza Younas**
- GitHub: [@Hamzaviour](https://github.com/Hamzaviour)
- LinkedIn: [Hamza Younas](https://linkedin.com/in/hamza-younas)
- Email: hamzavelous@gmail.com

---

🌲 **Built with ❤️ using Streamlit and Machine Learning | Elevvo Pathways**
