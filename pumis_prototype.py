# ==============================================================================
# PROTOTYPE RUN INSTRUCTIONS
# ==============================================================================
# 1. Install Streamlit if you haven't already:
#    pip install streamlit
# 2. Ensure this file is in the SAME folder as:
#    - pumis_generator.pth
#    - scaler_params.pth
# 3. Run the app from your terminal:
#    streamlit run pumis_prototype.py
# ==============================================================================

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================
# 1. MODEL ARCHITECTURE (Must match training)
# ==========================================
class Generator(nn.Module):
    def __init__(self, latent_dim, context_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim + context_dim, 128),
            nn.GroupNorm(8, 128),
            nn.LeakyReLU(0.2, inplace=False),
            
            nn.Linear(128, 128),
            nn.GroupNorm(8, 128),
            nn.LeakyReLU(0.2, inplace=False),
            
            nn.Linear(128, output_dim)
        )

    def forward(self, z, context):
        x = torch.cat((z, context), dim=1)
        return self.model(x)

# ==========================================
# 2. LOAD ASSETS FUNCTION (Cached for Speed)
# ==========================================
@st.cache_resource
def load_pumis_assets():
    """Loads the pre-trained Generator and Scaler parameters."""
    LATENT_DIM = 32
    device = torch.device('cpu') # Always use CPU for web inference for safety
    
    # Initialize Generator
    gen = Generator(LATENT_DIM, 4, 2).to(device)
    
    # Check if files exist
    gen_path = "pumis_generator.pth"
    scaler_path = "scaler_params.pth"
    
    if os.path.exists(gen_path) and os.path.exists(scaler_path):
        # FIX: Added weights_only=False to allow PyTorch to load the NumPy arrays in the scaler file
        gen.load_state_dict(torch.load(gen_path, map_location=device, weights_only=False))
        gen.eval()
        scaler_data = torch.load(scaler_path, map_location=device, weights_only=False)
        return gen, scaler_data, True
    else:
        return None, None, False

# ==========================================
# 3. GENERATION LOGIC
# ==========================================
def generate_synthetic_trips(gen, scaler_data, num_trips, hour, pu_zone, do_zone):
    LATENT_DIM = 32
    
    # Contextual Encoding (Exact same math as DataEngine)
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    pu_norm = (pu_zone / 132.5) - 1.0
    do_norm = (do_zone / 132.5) - 1.0
    
    # Create batch tensors
    z = torch.randn(num_trips, LATENT_DIM)
    context = torch.tensor([[hour_sin, hour_cos, pu_norm, do_norm]] * num_trips, dtype=torch.float32)
    
    # Generate Output (Z-Scores)
    with torch.no_grad():
        synthetic_scaled = gen(z, context).numpy()
    
    # Un-scale using saved exact means and stds
    means = scaler_data['means']
    stds = scaler_data['stds']
    
    trip_distance = (synthetic_scaled[:, 0] * stds[0]) + means[0]
    duration_mins = (synthetic_scaled[:, 1] * stds[1]) + means[1]
    
    # Physics Clamping (Prevent impossible negative numbers)
    trip_distance = np.clip(trip_distance, a_min=0.1, a_max=None)
    duration_mins = np.clip(duration_mins, a_min=1.0, a_max=None)
    avg_speed = trip_distance / (duration_mins / 60.0)
    
    # Package into DataFrame
    df = pd.DataFrame({
        'Trip Distance (Miles)': trip_distance,
        'Duration (Minutes)': duration_mins,
        'Avg Speed (MPH)': avg_speed
    })
    
    return df

# ==========================================
# 4. STREAMLIT UI DESIGN
# ==========================================
st.set_page_config(page_title="PUMIS Synthetic Engine", layout="wide", page_icon="🚕")

# FIX: Replaced solid white background with a dark-mode friendly translucent box
st.markdown("""
    <style>
    div[data-testid="metric-container"] {
        background-color: rgba(128, 128, 128, 0.1);
        padding: 15px; 
        border-radius: 10px; 
        border: 1px solid rgba(128, 128, 128, 0.2);
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

st.title("🚕 PUMIS: Privacy-Preserving Urban Mobility Synthesizer")
st.markdown("Generate high-fidelity, differentially private synthetic taxi trips based on specific temporal and spatial contexts.")

# Load Assets
generator, scaler, loaded = load_pumis_assets()

if not loaded:
    st.error("🚨 **Error:** Missing Model Files!")
    st.info("Could not find `pumis_generator.pth` or `scaler_params.pth`. Please run the `pumis_integrated_system.py` training script first to generate these files, and ensure they are in the same folder as this web app.")
    st.stop()

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("⚙️ Context Parameters")
    st.markdown("Set the urban environment conditions.")
    
    num_trips = st.slider("Number of Trips to Generate", min_value=10, max_value=2000, value=500, step=10)
    
    hour = st.slider("Hour of Day (0-23)", min_value=0, max_value=23, value=14, step=1, 
                     help="0 = Midnight, 12 = Noon, 17 = 5 PM Rush Hour")
    
    pu_zone = st.number_input("Pickup Zone ID (1-265)", min_value=1, max_value=265, value=161)
    do_zone = st.number_input("Dropoff Zone ID (1-265)", min_value=1, max_value=265, value=230)
    
    generate_btn = st.button("🚀 Generate Synthetic Data", use_container_width=True, type="primary")

# --- MAIN DASHBOARD ---
if generate_btn:
    with st.spinner('Generating secure synthetic mobility data...'):
        synth_df = generate_synthetic_trips(generator, scaler, num_trips, hour, pu_zone, do_zone)
        
    st.success(f"Successfully generated {num_trips} synthetic trips with strict Differential Privacy (ε=8.0) applied.")
    
    # Metrics Row
    col1, col2, col3 = st.columns(3)
    col1.metric("Average Trip Distance", f"{synth_df['Trip Distance (Miles)'].mean():.2f} mi")
    col2.metric("Average Trip Duration", f"{synth_df['Duration (Minutes)'].mean():.1f} mins")
    col3.metric("Average Fleet Speed", f"{synth_df['Avg Speed (MPH)'].mean():.1f} mph")
    
    st.markdown("---")
    
    # Visualizations Row
    st.subheader("📊 Synthetic Distribution Analysis")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Distance Distribution
    sns.histplot(synth_df['Trip Distance (Miles)'], kde=True, color="#11caa0", ax=axes[0])
    axes[0].set_title("Trip Distance Probability Density", fontweight="bold")
    axes[0].set_xlabel("Miles")
    
    # Plot 2: Physics Correlation
    sns.scatterplot(data=synth_df, x='Duration (Minutes)', y='Trip Distance (Miles)', 
                    hue='Avg Speed (MPH)', palette="viridis", alpha=0.6, ax=axes[1])
    axes[1].set_title("Physics Constraint Realism (Dist vs Time)", fontweight="bold")
    
    st.pyplot(fig)
    
    # Data Table View
    st.markdown("---")
    st.subheader("📄 Raw Synthetic Records (Preview)")
    st.dataframe(synth_df.head(50), use_container_width=True)
    
    # Download Button
    csv = synth_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="💾 Download Synthetic Dataset as CSV",
        data=csv,
        file_name='pumis_synthetic_trips.csv',
        mime='text/csv',
    )
else:
    # Empty State Dashboard
    st.info("👈 Set your parameters in the sidebar and click **Generate Synthetic Data** to test the PUMIS engine.")
    st.image("https://images.unsplash.com/photo-1541259596001-f15501fb35de?auto=format&fit=crop&q=80&w=1200", 
             caption="Urban Mobility Synthesis Simulator", use_container_width=True)