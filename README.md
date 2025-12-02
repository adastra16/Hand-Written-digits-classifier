# Handwritten Digit Classifier

A Streamlit web application that classifies handwritten digits using a neural network trained on the MNIST dataset.
Deployed link: https://hand-written-digits-classifier-ddycucz5jn36wwdxrr5q2o.streamlit.app/

## Features

- 🎨 **Draw Digits**: Draw digits directly in the browser and get instant predictions
- 📤 **Upload Images**: Upload images of handwritten digits for classification
- 📊 **Model Training**: View training metrics and confusion matrix
- 📈 **Confidence Scores**: See prediction probabilities for each digit
- 🎯 **High Accuracy**: ~97.8% accuracy on test set

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Setup

1. Clone the repository:
\`\`\`bash
git clone https://github.com/yourusername/digit-classifier.git
cd digit-classifier
\`\`\`

2. Create a virtual environment (optional but recommended):
\`\`\`bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
\`\`\`

3. Install dependencies:
\`\`\`bash
pip install -r requirements.txt
\`\`\`

## Usage

Run the Streamlit app:
\`\`\`bash
streamlit run app.py
\`\`\`

The app will open in your browser at `http://localhost:8501`

## Deployment on Streamlit Cloud

1. Push your code to GitHub
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Click "New app"
4. Select your repository and branch
5. Set the main file path to `app.py`
6. Click "Deploy"

## Project Structure

\`\`\`
digit-classifier/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── .gitignore            # Git ignore file
└── digit_model.h5        # Trained model (generated on first run)
\`\`\`

## Model Architecture

\`\`\`
Input (28×28 images)
    ↓
Flatten Layer (784 neurons)
    ↓
Dense Layer (100 neurons, ReLU activation)
    ↓
Dense Layer (10 neurons, Softmax activation)
    ↓
Output (digit prediction 0-9)
\`\`\`

## Dataset

- **MNIST**: 70,000 images of handwritten digits
- **Training Set**: 60,000 images
- **Test Set**: 10,000 images
- **Image Size**: 28×28 pixels (grayscale)

## Performance

- **Test Accuracy**: ~97.8%
- **Test Loss**: ~0.081

## Technologies

- **TensorFlow/Keras**: Deep learning framework
- **Streamlit**: Web application framework
- **NumPy**: Numerical computing
- **Matplotlib & Seaborn**: Data visualization
- **PIL**: Image processing

## License

This project is open source and available under the MIT License.

