## ❤️ Heart Disease Prediction Analysis

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)
![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)

A beautiful and intuitive **AI-Powered Medical Analysis System** for predicting heart disease using machine learning algorithms. This desktop application features a modern GUI built with Tkinter and employs multiple ML models including K-Nearest Neighbors (KNN) and Random Forest classifiers.

## 📸 Screenshots

### Main Interface

![Application Interface](https://github.com/professionalman/Heart_disease_prediction/blob/main/screenshots/Picture1.png)

### Data Visualization

![Data Visualization](https://github.com/professionalman/Heart_disease_prediction/blob/main/screenshots/Picture2.png)

### Model Training

![Model Training](https://github.com/professionalman/Heart_disease_prediction/blob/main/screenshots/Picture5.png)

## ✨ Features

### 📊 **Data Explorer**

- 📂 Load CSV datasets with file dialog
- 📈 View comprehensive dataset statistics
- 🔍 Inspect data types and information
- ✅ Automatic data preprocessing and normalization

### 📉 **Data Visualizer**

- 🔥 **Correlation Heatmap** - Visualize feature relationships
- 📈 **Feature Distribution** - Histogram analysis of all features
- 🎯 **Target Distribution** - Class balance visualization
- 🎨 Modern, professional chart styling

### 🧠 **Model Trainer**

- 🔍 **K-Nearest Neighbors (KNN)** - Test multiple K values (1-20)
- 🌳 **Random Forest Classifier** - Ensemble learning approach
- 📊 10-fold cross-validation for robust evaluation
- 📈 Real-time performance visualization

### 🎨 **Modern UI/UX**

- 💎 Beautiful card-based design
- 🌈 Professional color scheme
- 📱 Responsive layout (1400x900)
- ✨ Smooth animations and hover effects
- 🎯 Intuitive navigation with tabbed interface

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/heart-disease-prediction.git
   cd heart-disease-prediction
   ```

2. **Install required packages**

   ```bash
   pip install -r requirements.txt
   ```

   Or install manually:

   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn
   ```

3. **Run the application**
   ```bash
   python Heart_.py
   ```

## 📋 Dataset Requirements

The application expects a CSV file with the following features:

| Feature    | Description                       | Type        |
| ---------- | --------------------------------- | ----------- |
| `age`      | Age in years                      | Integer     |
| `sex`      | Sex (1 = male, 0 = female)        | Binary      |
| `cp`       | Chest pain type (0-3)             | Categorical |
| `trestbps` | Resting blood pressure (mm Hg)    | Integer     |
| `chol`     | Serum cholesterol (mg/dl)         | Integer     |
| `fbs`      | Fasting blood sugar > 120 mg/dl   | Binary      |
| `restecg`  | Resting ECG results (0-2)         | Categorical |
| `thalach`  | Maximum heart rate achieved       | Integer     |
| `exang`    | Exercise induced angina           | Binary      |
| `oldpeak`  | ST depression induced by exercise | Float       |
| `slope`    | Slope of peak exercise ST segment | Categorical |
| `ca`       | Number of major vessels (0-4)     | Integer     |
| `thal`     | Thalassemia (0-3)                 | Categorical |
| `target`   | Heart disease (1) or not (0)      | Binary      |

### Sample Dataset

A sample dataset (`heart_disease_data.csv`) with 303 patient records is included for testing purposes.

## 🛠️ Technologies Used

### Core Technologies

- **Python 3.x** - Programming language
- **Tkinter** - GUI framework
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation and analysis
- **Matplotlib** - Data visualization
- **Seaborn** - Statistical data visualization

### Machine Learning

- **Scikit-learn** - ML algorithms and tools
  - K-Nearest Neighbors (KNN)
  - Random Forest Classifier
  - StandardScaler for normalization
  - Cross-validation scoring

## 📖 How to Use

### 1️⃣ Load Dataset

1. Open the application
2. Navigate to the **📂 Data Explorer** tab
3. Click **"📂 Load Dataset"**
4. Select your CSV file
5. View dataset statistics and information

### 2️⃣ Visualize Data

1. Go to the **📊 Data Visualizer** tab
2. Choose from available visualizations:
   - **Correlation Heatmap** - See feature correlations
   - **Feature Distribution** - Analyze feature patterns
   - **Target Distribution** - Check class balance

### 3️⃣ Train Models

1. Switch to the **🧠 Model Trainer** tab
2. Click **"🔍 Evaluate KNN"** to test KNN classifier
   - Evaluates K values from 1 to 20
   - Displays performance graph
   - Shows optimal K value
3. Click **"🌳 Evaluate Random Forest"** to test RF classifier
   - Shows cross-validation score
   - Displays accuracy percentage

## 🎯 Model Performance

### K-Nearest Neighbors (KNN)

- **Algorithm**: Instance-based learning
- **Evaluation**: 10-fold cross-validation
- **Parameters**: K values from 1 to 20
- **Output**: Performance graph and optimal K value

### Random Forest

- **Algorithm**: Ensemble learning (10 trees)
- **Evaluation**: 10-fold cross-validation
- **Output**: Average accuracy score

## 📁 Project Structure

```
heart-disease-prediction/
│
├── Heart_.py                 # Main application file
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── heart_disease_data.csv    # Sample dataset (optional)
│
└── screenshots/              # Application screenshots
    ├── main_interface.png
    ├── visualization.png
    └── model_training.png
```

## 🔧 Configuration

The application uses these default settings:

- **Window Size**: 1400x900 pixels
- **Theme**: Modern flat design
- **Colors**:
  - Primary: `#3498db` (Blue)
  - Success: `#27ae60` (Green)
  - Danger: `#e74c3c` (Red)
- **Font**: Segoe UI

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 Future Enhancements

- [ ] Add more ML algorithms (SVM, Neural Networks, etc.)
- [ ] Implement model comparison dashboard
- [ ] Add prediction feature for new patients
- [ ] Export model results to PDF/CSV
- [ ] Add data preprocessing options
- [ ] Implement feature importance analysis
- [ ] Add confusion matrix visualization
- [ ] Support for more dataset formats (Excel, JSON)
- [ ] Add dark mode theme
- [ ] Implement model saving/loading

## 🐛 Known Issues

- None currently reported

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Your Name**

- GitHub: [Kavya Arora](https://github.com/professionalman)
- Email: 24mca20091@cuchd.in

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the heart disease dataset
- Scikit-learn documentation and community
- Python data science community
- All contributors and testers

## 📧 Contact

For questions, suggestions, or issues, please open an issue on GitHub or contact the author directly.

---

<p align="center">
  Made with ❤️ and Python
</p>

<p align="center">
  <sub>Built as part of MCA SEM 3 Machine Learning Project</sub>
</p>"
