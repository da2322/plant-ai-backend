from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import io
import base64
import matplotlib.cm as cm
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)
# Allow your HTML page to communicate with this API
CORS(app) 

# --- 1. LOAD MODEL & CLASSES ---
model = tf.keras.models.load_model('finetuned_plant_model.keras')

CLASS_NAMES = [
    'Grape_Black_Rot', 'Grape_Esca_Black_Measles', 'Grape_Healthy', 
    'Grape_Leaf_Blight', 'Potato_Early_Blight', 'Potato_Healthy', 
    'Potato_Late_Blight', 'Tomato_Early_Blight', 'Tomato_Healthy', 
    'Tomato_Late_Blight', 'Tomato_Yellow_Leaf_Curl_Virus'
]

# --- 2. GRAD-CAM ALGORITHM ---
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="out_relu"):
    base_model = None
    base_model_idx = -1
    for i, layer in enumerate(model.layers):
        if isinstance(layer, tf.keras.Model):
            base_model = layer
            base_model_idx = i
            break

    if base_model is None:
        raise ValueError("Could not find nested base model!")

    inner_grad_model = tf.keras.models.Model(
        inputs=base_model.inputs,
        outputs=[base_model.get_layer(last_conv_layer_name).output, base_model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, x = inner_grad_model(img_array)
        tape.watch(last_conv_layer_output)
        for layer in model.layers[base_model_idx + 1:]:
            x = layer(x) 
        preds = x
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index] 

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# --- 3. OVERLAY FUNCTION ---
def create_gradcam_overlay(img_pil, heatmap, alpha=0.5):
    img = np.array(img_pil)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    import matplotlib
    jet = matplotlib.colormaps["jet"]
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap] * 255
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    return superimposed_img

# --- 4. API ENDPOINT ---
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No image file uploaded'}), 400
        
    file = request.files['file']
    img_pil = Image.open(file.stream).convert('RGB')
    
    # Preprocess
    img_resized = img_pil.resize((224, 224), resample=Image.Resampling.NEAREST)
    img_array = keras_image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Predict
    predictions = model.predict(img_array, verbose=0)[0]
    top_index = int(np.argmax(predictions))
    top_guess = CLASS_NAMES[top_index]
    confidence = float(predictions[top_index] * 100)
    all_probs = predictions.tolist()

    # Generate Heatmap
    heatmap = make_gradcam_heatmap(img_array, model)
    overlay_img_array = create_gradcam_overlay(img_pil, heatmap)
    
    # Convert GradCAM image to Base64 so HTML can display it
    overlay_pil = Image.fromarray(overlay_img_array)
    buffered = io.BytesIO()
    overlay_pil.save(buffered, format="JPEG")
    gradcam_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return jsonify({
        'topClass': top_guess,
        'confidence': confidence,
        'allProbs': all_probs,
        'gradcamImage': f"data:image/jpeg;base64,{gradcam_b64}"
    })

if __name__ == '__main__':
    app.run(port=5000, debug=True)
