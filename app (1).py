import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.cm as cm

# --- 1. PRO PAGE SETUP (Wide Layout) ---
st.set_page_config(page_title="Neuro-AI Diagnostic Tool", page_icon="🧠", layout="wide")

st.markdown("""
    <style>
    .main-title { font-size: 3rem; font-weight: bold; color: #1E3A8A; text-align: center; margin-bottom: 0px;}
    .sub-title { font-size: 1.2rem; color: #64748B; text-align: center; margin-bottom: 2rem;}
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🧠 Neuro-AI Diagnostic Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Advanced MRI Classification & XAI Tumor Localization</p>', unsafe_allow_html=True)
st.divider()

# --- 2. CACHING THE MODEL ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('brain_tumor_model.h5')

model = load_model()
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

last_conv_layer_name = None
for layer in reversed(model.layers):
    if isinstance(layer, tf.keras.layers.Conv2D):
        last_conv_layer_name = layer.name
        break

# --- 3. GRAD-CAM ENGINE ---
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    last_conv_layer = model.get_layer(last_conv_layer_name)
    last_conv_layer_model = tf.keras.Model(model.inputs, last_conv_layer.output)

    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    layer_index = model.layers.index(last_conv_layer)
    for layer in model.layers[layer_index + 1:]:
        x = layer(x)
    classifier_model = tf.keras.Model(classifier_input, x)

    with tf.GradientTape() as tape:
        last_conv_layer_output = last_conv_layer_model(img_array)
        tape.watch(last_conv_layer_output)
        preds = classifier_model(last_conv_layer_output)
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# --- 4. SIDEBAR MENU ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063223.png", width=100)
    st.title("Patient Input")
    st.info("Please upload a T1-weighted or T2-weighted Brain MRI scan for analysis.")
    uploaded_file = st.file_uploader("Upload MRI...", type=["jpg", "png", "jpeg"])

    st.markdown("---")
    st.markdown("**AI Capabilities:**")
    st.markdown("- Glioma Detection")
    st.markdown("- Meningioma Detection")
    st.markdown("- Pituitary Detection")
    st.markdown("- Healthy Brain Verification")

# --- 5. MAIN DASHBOARD LOGIC ---
if uploaded_file is None:
    st.warning("👈 Please upload an MRI scan from the sidebar menu to begin analysis.")
else:
    with st.spinner("Analyzing MRI Scan and Generating Heatmap..."):

        image = Image.open(uploaded_file).convert('RGB')
        img_resized = image.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
        img_array_batch = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array_batch)
        predicted_class = class_names[np.argmax(predictions[0])]
        confidence = np.max(predictions[0]) * 100

        heatmap = make_gradcam_heatmap(img_array_batch, model, last_conv_layer_name)
        heatmap_rescaled = np.uint8(255 * heatmap)
        jet = cm.jet(np.arange(256))[:, :3]
        jet_heatmap = jet[heatmap_rescaled]
        jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((img_array.shape[1], img_array.shape[0]))
        jet_heatmap_array = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

        superimposed_img = jet_heatmap_array * 0.4 + img_array
        superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

    # --- 6. DISPLAYING RESULTS BEAUTIFULLY ---
    if predicted_class == 'notumor':
        st.success(f"✅ **Diagnosis:** No Tumor Detected (Confidence: {confidence:.2f}%)")
    else:
        st.error(f"⚠️ **Diagnosis:** {predicted_class.capitalize()} Tumor Detected (Confidence: {confidence:.2f}%)")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("<h4 style='text-align: center;'>Original MRI</h4>", unsafe_allow_html=True)
        # Fix: Changed use_container_width to width="stretch"
        st.image(image, width="stretch")

    with col2:
        st.markdown("<h4 style='text-align: center;'>AI Heatmap (Grad-CAM)</h4>", unsafe_allow_html=True)
        # Fix: Now passing jet_heatmap instead of raw heatmap
        st.image(jet_heatmap, width="stretch")

    with col3:
        st.markdown("<h4 style='text-align: center;'>Superimposed Result</h4>", unsafe_allow_html=True)
        # Fix: Changed use_container_width to width="stretch"
        st.image(superimposed_img, width="stretch")
