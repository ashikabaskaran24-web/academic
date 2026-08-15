from flask import Flask, render_template, request
import tensorflow as tf
print(tf.__version__)
model=tf.keras.models.load_model("leaf_disease_model.keras")
import numpy as np
import os

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("leaf_disease_model.keras")

# Class labels
CLASS_NAMES = [
    'bacterial_leaf_blight',
    'brown_spot',
    'healthy',
    'leaf_blast',
    'leaf_scald',
    'narrow_brown_spot'
]

# Tamil Translations
TRANSLATIONS = {
    "bacterial_leaf_blight": "பாக்டீரியா இலை வாடுதல்",
    "brown_spot": "பழுப்பு புள்ளி நோய்",
    "healthy": "ஆரோக்கியமான இலை",
    "leaf_blast": "இலை வெடிப்பு நோய்",
    "leaf_scald": "இலை சுடுதல் நோய்",
    "narrow_brown_spot": "மெல்லிய பழுப்பு புள்ளி நோய்"
}

# Treatment and fertilizer info (English)
TREATMENTS = {
    "bacterial_leaf_blight": {
        "treatment": "Use copper-based fungicides and avoid water stagnation.",
        "fertilizer": "Apply potash-rich fertilizers to strengthen the plant."
    },
    "brown_spot": {
        "treatment": "Spray Mancozeb or Carbendazim fungicide.",
        "fertilizer": "Use nitrogen and potassium fertilizers moderately."
    },
    "healthy": {
        "treatment": "No treatment needed. Maintain good irrigation and nutrition.",
        "fertilizer": "Balanced NPK fertilizers are recommended."
    },
    "leaf_blast": {
        "treatment": "Use Tricyclazole fungicide and avoid excess nitrogen.",
        "fertilizer": "Apply phosphorus fertilizers to promote resistance."
    },
    "leaf_scald": {
        "treatment": "Apply copper oxychloride spray and ensure proper drainage.",
        "fertilizer": "Use organic compost and avoid over-irrigation."
    },
    "narrow_brown_spot": {
        "treatment": "Spray Propiconazole or Tricyclazole.",
        "fertilizer": "Maintain balanced NPK level for the crop."
    }
}

# Tamil translations for treatment and fertilizer
TREATMENTS_TAMIL = {
    "bacterial_leaf_blight": {
        "treatment": "காப்பர் அடிப்படையிலான பூஞ்சைநாசினி பயன்படுத்தவும் மற்றும் நீர் தேக்கம் தவிர்க்கவும்.",
        "fertilizer": "தாவரத்தை வலுப்படுத்த பொட்டாசியம் அதிகம் உள்ள உரங்களை பயன்படுத்தவும்."
    },
    "brown_spot": {
        "treatment": "மாங்கோசெப் அல்லது கார்பெண்டசிம் பூஞ்சைநாசினி தெளிக்கவும்.",
        "fertilizer": "நைட்ரஜன் மற்றும் பொட்டாசியம் உரங்களை மிதமான அளவில் பயன்படுத்தவும்."
    },
    "healthy": {
        "treatment": "சிகிச்சை தேவையில்லை. நல்ல பாசனம் மற்றும் சத்துணவு பராமரிக்கவும்.",
        "fertilizer": "சமநிலை NPK உரங்களை பயன்படுத்த பரிந்துரைக்கப்படுகிறது."
    },
    "leaf_blast": {
        "treatment": "ட்ரிசைக்ளசோல் பூஞ்சைநாசினி பயன்படுத்தவும் மற்றும் அதிக நைட்ரஜன் தவிர்க்கவும்.",
        "fertilizer": "எதிர்ப்பு சக்தி அதிகரிக்க பாஸ்பரஸ் உரங்கள் பயன்படுத்தவும்."
    },
    "leaf_scald": {
        "treatment": "காப்பர் ஆக்சிகுளோரைடு தெளிக்கவும் மற்றும் சரியான வடிகால் உறுதி செய்யவும்.",
        "fertilizer": "உயிர்ச்சத்து கூழ் பயன்படுத்தவும் மற்றும் அதிக பாசனம் தவிர்க்கவும்."
    },
    "narrow_brown_spot": {
        "treatment": "ப்ரோப்பிகோனசோல் அல்லது ட்ரிசைக்ளசோல் தெளிக்கவும்.",
        "fertilizer": "பயிரின் சமநிலை NPK அளவை பராமரிக்கவும்."
    }
}


@app.route('/')
def home():
    return render_template('index.html')
def estimate_yield(disease, severity):
    if disease == "healthy":
        return {
            "yield_percent": "100%",
            "loss": "0%",
            "message": "No yield loss expected"
        }

    if severity == "Low":
        return {
            "yield_percent": "85–90%",
            "loss": "10–15%",
            "message": "Minor yield reduction"
        }

    elif severity == "Medium":
        return {
            "yield_percent": "60–70%",
            "loss": "30–40%",
            "message": "Moderate yield loss"
        }

    else:  # High
        return {
            "yield_percent": "35–50%",
            "loss": "50–65%",
            "message": "Severe yield loss expected"
        }


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', prediction_text="No image selected!")

    basepath = os.path.dirname(__file__)
    file_path = os.path.join(basepath, 'uploads', file.filename)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    file.save(file_path)

    # Preprocess
    from tensorflow.keras.preprocessing import image
    img = image.load_img(file_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    confidence = float(np.max(prediction)) * 100
    class_index = np.argmax(prediction)
    disease_name = CLASS_NAMES[class_index]

    tamil_name = TRANSLATIONS[disease_name]

    if disease_name == "healthy":
     severity_level = "No disease detected"
     treatment = "No treatment required. The leaf is healthy."
     fertilizer = "Continue balanced NPK fertilizer and proper irrigation."

     treatment_tamil = "சிகிச்சை தேவையில்லை. இலை ஆரோக்கியமாக உள்ளது."
     fertilizer_tamil = "சமநிலை NPK உரமும் சரியான பாசனமும் தொடரவும்."
    else:
     
     if confidence < 60:
      severity_level = "Low"
     elif confidence < 85:
      severity_level = "Medium"
     else:
      severity_level = "High"

    treatment = TREATMENTS[disease_name]["treatment"]
    fertilizer = TREATMENTS[disease_name]["fertilizer"]

    treatment_tamil = TREATMENTS_TAMIL[disease_name]["treatment"]
    fertilizer_tamil = TREATMENTS_TAMIL[disease_name]["fertilizer"]
    yield_info = estimate_yield(disease_name, severity_level)

    return render_template(
        'index.html',
        disease=disease_name.replace('_', ' ').title(),
        tamil_name=tamil_name,
        severity=severity_level,
        treatment=treatment,
        fertilizer=fertilizer,
        treatment_tamil=treatment_tamil,
        fertilizer_tamil=fertilizer_tamil,
        confidence=f"{confidence:.2f}%",
        yield_percent=yield_info["yield_percent"],
        yield_loss=yield_info["loss"],
        yield_message=yield_info["message"]
        
    )


if __name__ == "__main__":
    app.run(debug=True)
