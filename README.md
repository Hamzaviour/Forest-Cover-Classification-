# 🌲 Forest Cover Classification Dashboard

A professional Streamlit application for forest cover type classification using advanced machine learning algorithms.

**Created by Hamza Younas | Elevvo Pathways**

![Forest Cover Classification](https://img.shields.io/badge/ML-Forest%20Classification-green)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)

## 🚀 Live Demo

🌐 **[Try the Live Application](https://your-streamlit-app-url.streamlit.app)** *(Update with your deployed URL)*

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

## 🚀 Deployment

### Deploy to Streamlit Cloud
1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Deploy with one click!

### Deploy to Heroku
```bash
# Create Procfile
echo "web: sh setup.sh && streamlit run app.py" > Procfile

# Create setup.sh
echo "mkdir -p ~/.streamlit/
echo \"[server]
headless = true
port = \$PORT
enableCORS = false
\" > ~/.streamlit/config.toml" > setup.sh
```

## 📊 Screenshots

*Add screenshots of your application here*

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Elevvo Pathways** for project support
- UCI Machine Learning Repository for the Forest Cover Type dataset
- Streamlit team for the amazing framework

## 📧 Contact

**Hamza Younas**
- GitHub: [@your-github-username](https://github.com/your-github-username)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/your-profile)
- Email: your.email@example.com

---

🌲 **Built with ❤️ using Streamlit and Machine Learning | Elevvo Pathways**
