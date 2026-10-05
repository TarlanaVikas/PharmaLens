# PharmaLens

An AI-powered healthcare application that identifies medicines from images and provides relevant healthcare insights — using computer vision and deep learning to make medication information instantly accessible.

---

## 🧠 How It Works

```
User uploads medicine image
          ↓
OpenCV preprocesses and analyzes image
          ↓
TensorFlow model classifies medicine
          ↓
System returns: Name · Dosage · Uses · Side Effects
```

---

## ✨ Features

- **Medicine identification** from images using deep learning
- **Computer vision-based recognition** — works from photos of packaging or pills
- **Healthcare insights** — returns dosage, uses, and side effects for identified medicine
- **Web-based interface** — no installation needed for end users
- **Automated image analysis** and classification pipeline
- **Practical healthcare assistance** through AI-powered predictions

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| **Web Framework** | Flask |
| **Computer Vision** | OpenCV |
| **Deep Learning** | TensorFlow |
| **Language** | Python |

---

## ⚙️ Setup & Installation

```bash
# 1. Clone the repository
git clone https://github.com/prathamparmar1/Medicine-Classification-Using-ML.git
cd Medicine-Classification-Using-ML

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Flask app
python app.py
```

The app will be available at `http://localhost:5000`

---

## 🚀 Usage

1. Open the web interface at `http://localhost:5000`
2. Upload an image of a medicine (packaging or pill)
3. The system will return:
   - Medicine name
   - Recommended dosage
   - Common uses
   - Known side effects


## ⚠️ Disclaimer

This application is for educational purposes only. Always consult a qualified healthcare professional before making any medical decisions.

---
# How to Operate

OPEN PROJECT: 
    1. Run file extractfeatures.py
	2. Run file app.py
	3. View at http://127.0.0.1:5000 (localhost on any browser)

USE PROJECT:
	1. As Home page appears Click on "Choose File"
	2. You can choose any image from your local system by clicking on "choose an image"
	3. Select Image and click on upload photo

RESULT:
	1. Result appears after loading
	2. If not satisfied try with another Image.


