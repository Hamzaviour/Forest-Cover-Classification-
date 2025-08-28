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

<img width="3000" height="1800" alt="Feature_Importance_xgb" src="https://github.com/user-attachments/assets/0eba258a-3898-4918-b5a1-41eaf5aa72c7" />
<img width="3000" height="1800" alt="Feature_Importance_rf" src="https://github.com/user-attachments/assets/4e17a225-889a-4d1e-95e3-b8299a7b873d" />
<img width="3000" height="1800" alt="Feature_Importance_dt" src="https://github.com/user-attachments/assets/6f6eb0eb-e475-4317-bd1f-fffc82e70141" />
<img width="2400" height="1500" alt="distribution_cover_types" src="https://github.com/user-attachments/assets/c76fbfae-d002-4509-b5cb-f7cba2fef7a5" />
<img width="1920" height="1440" alt="Confusion_Matrix_Xgb" src="https://github.com/user-attachments/assets/aa9ec5d1-d684-407d-b5f5-9a5a228fdce8" />
<img width="1920" height="1440" alt="confusion_matrix_rf" src="https://github.com/user-attachments/assets/a3d3d0f6-fd64-4e48-8eb2-5f61f516076b" />
<img width="1920" height="1440" alt="confusion_matrix_dt" src="https://github.com/user-attachments/assets/4ca19fb0-d73d-4b86-b052-ddfe63b6ace3" />
<img width="3000" height="1800" alt="model_comparison" src="https://github.com/user-attachments/assets/79a4bf85-9ee0-45f6-8a1d-eebd37e15975" />




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
