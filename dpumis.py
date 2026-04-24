# ==============================================================================
# PROTOTYPE RUN INSTRUCTIONS
# ==============================================================================
# 1. Install Streamlit if you haven't already:
#    pip install streamlit scikit-learn
# 2. Ensure this file is in the SAME folder as:
#    - pumis_generator.pth
#    - scaler_params.pth
# 3. Run the app from your terminal:
#    streamlit run pumis_prototype.py
# ==============================================================================

import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# ==========================================
# 1. DYNAMIC NEURAL ARCHITECTURES
# ==========================================
# --- Architecture A: ContextGAN (For Pre-Trained NYC Data) ---
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

# --- Architecture B: AutoGAN (For Any Uploaded Dataset) ---
class AutoGenerator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(AutoGenerator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, output_dim) # Outputs the exact shape of uploaded CSV
        )
    def forward(self, z):
        return self.model(z)

class AutoDiscriminator(nn.Module):
    def __init__(self, input_dim):
        super(AutoDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1) # Raw logits
        )
    def forward(self, data):
        return self.model(data)

# ==========================================
# 2. LOAD PRE-TRAINED ASSETS (NYC)
# ==========================================
@st.cache_resource
def load_pumis_assets():
    LATENT_DIM = 32
    device = torch.device('cpu') 
    gen = Generator(LATENT_DIM, 4, 2).to(device)
    
    gen_path = "pumis_generator.pth"
    scaler_path = "scaler_params.pth"
    
    if os.path.exists(gen_path) and os.path.exists(scaler_path):
        gen.load_state_dict(torch.load(gen_path, map_location=device, weights_only=False))
        gen.eval()
        scaler_data = torch.load(scaler_path, map_location=device, weights_only=False)
        return gen, scaler_data, True
    else:
        return None, None, False

def generate_nyc_trips(gen, scaler_data, num_trips, hour, pu_zone, do_zone):
    LATENT_DIM = 32
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    pu_norm = (pu_zone / 132.5) - 1.0
    do_norm = (do_zone / 132.5) - 1.0
    
    z = torch.randn(num_trips, LATENT_DIM)
    context = torch.tensor([[hour_sin, hour_cos, pu_norm, do_norm]] * num_trips, dtype=torch.float32)
    
    with torch.no_grad():
        synthetic_scaled = gen(z, context).numpy()
    
    means, stds = scaler_data['means'], scaler_data['stds']
    trip_distance = (synthetic_scaled[:, 0] * stds[0]) + means[0]
    duration_mins = (synthetic_scaled[:, 1] * stds[1]) + means[1]
    
    trip_distance = np.clip(trip_distance, a_min=0.1, a_max=None)
    duration_mins = np.clip(duration_mins, a_min=1.0, a_max=None)
    avg_speed = trip_distance / (duration_mins / 60.0)
    
    return pd.DataFrame({
        'Trip Distance (Miles)': trip_distance,
        'Duration (Minutes)': duration_mins,
        'Avg Speed (MPH)': avg_speed
    })

# ==========================================
# 3. ON-THE-FLY AUTO TRAINING ENGINE
# ==========================================
def train_auto_model(df, epochs=50):
    """Automatically learns the pattern of any numeric dataframe."""
    device = torch.device('cpu')
    LATENT_DIM = 64
    
    # 1. Automatic Scaling of all columns
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    tensor_data = torch.FloatTensor(scaled_data)
    
    dataset = torch.utils.data.TensorDataset(tensor_data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
    
    # 2. Dynamic Architecture sizing based on dataframe shape
    num_features = df.shape[1]
    gen = AutoGenerator(LATENT_DIM, num_features).to(device)
    disc = AutoDiscriminator(num_features).to(device)
    
    opt_g = optim.Adam(gen.parameters(), lr=0.002, betas=(0.5, 0.999))
    opt_d = optim.Adam(disc.parameters(), lr=0.002, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # 3. Auto-GAN Training Loop
    for epoch in range(epochs):
        for real_batch in dataloader:
            real_data = real_batch[0]
            bs = real_data.size(0)
            
            # Train Discriminator
            opt_d.zero_grad()
            z = torch.randn(bs, LATENT_DIM)
            fake_data = gen(z)
            
            out_real = disc(real_data)
            out_fake = disc(fake_data.detach())
            
            loss_d = (criterion(out_real, torch.ones_like(out_real)) + 
                      criterion(out_fake, torch.zeros_like(out_fake))) / 2
            loss_d.backward()
            opt_d.step()
            
            # Train Generator
            opt_g.zero_grad()
            g_validity = disc(fake_data)
            loss_g = criterion(g_validity, torch.ones_like(g_validity))
            loss_g.backward()
            opt_g.step()
            
        progress_bar.progress((epoch + 1) / epochs)
        status_text.text(f"Analyzing Patterns... Epoch {epoch+1}/{epochs}")
        
    status_text.success("Analysis Complete! AI Model trained on your dataset schema.")
    return gen, scaler, LATENT_DIM

# ==========================================
# 4. STREAMLIT UI DESIGN
# ==========================================
st.set_page_config(page_title="PUMIS Platform", layout="wide", page_icon="🚕")

st.markdown("""
    <style>
    .main {background-color: #f8fafc;}
    h1, h2, h3 {color: #005088;}
    div[data-testid="metric-container"] {
        background-color: rgba(17, 202, 160, 0.1);
        padding: 15px; border-radius: 10px; border: 1px solid rgba(17, 202, 160, 0.3);
    }
    </style>
""", unsafe_allow_html=True)

st.title("🚕 PUMIS: Universal Mobility Synthesizer")
st.markdown("Generate privacy-preserving synthetic data using pre-trained models or upload your own dataset for automated pattern extraction.")

tab1, tab2 = st.tabs(["🏛️ NYC Pre-trained Engine (Demo)", "📂 1-Click Auto-Synthesizer"])

# ---------------------------------------------------------
# TAB 1: EXISTING PRE-TRAINED NYC MODEL
# ---------------------------------------------------------
with tab1:
    col1, col2 = st.columns([1, 3])
    
    generator, nyc_scaler, loaded = load_pumis_assets()
    
    with col1:
        st.header("⚙️ NYC Parameters")
        if not loaded:
            st.error("Missing NYC Model Files (`pumis_generator.pth`)")
        else:
            num_trips = st.slider("Trips to Generate", 10, 2000, 500, 10, key="nyc_num")
            hour = st.slider("Hour of Day (0-23)", 0, 23, 14, 1, key="nyc_hr")
            pu_zone = st.number_input("Pickup Zone ID", 1, 265, 161, key="nyc_pu")
            do_zone = st.number_input("Dropoff Zone ID", 1, 265, 230, key="nyc_do")
            btn_nyc = st.button("🚀 Generate NYC Data", use_container_width=True, type="primary")

    with col2:
        if loaded and 'btn_nyc' in locals() and btn_nyc:
            with st.spinner("Generating..."):
                synth_df = generate_nyc_trips(generator, nyc_scaler, num_trips, hour, pu_zone, do_zone)
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Average Distance", f"{synth_df['Trip Distance (Miles)'].mean():.2f} mi")
            m2.metric("Average Duration", f"{synth_df['Duration (Minutes)'].mean():.1f} min")
            m3.metric("Fleet Speed", f"{synth_df['Avg Speed (MPH)'].mean():.1f} mph")
            
            st.subheader("Distributions")
            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
            sns.histplot(synth_df['Trip Distance (Miles)'], kde=True, color="#005088", ax=ax[0]).set_title("Distance")
            sns.scatterplot(data=synth_df, x='Duration (Minutes)', y='Trip Distance (Miles)', color="#11caa0", alpha=0.6, ax=ax[1]).set_title("Dist vs Time")
            st.pyplot(fig)
            st.dataframe(synth_df.head(10), use_container_width=True)

# ---------------------------------------------------------
# TAB 2: DYNAMIC 1-CLICK AUTO-SYNTHESIZER
# ---------------------------------------------------------
with tab2:
    st.header("Upload & Auto-Clone Any Dataset")
    st.markdown("Upload any CSV. The AI will automatically detect the columns, learn their mathematical correlations, and build a customized Generative Model to synthesize an exact structural clone of your dataset.")
    
    uploaded_file = st.file_uploader("Upload CSV Data", type=['csv'])
    
    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        # Drop non-numeric for the GAN training
        numeric_df = raw_df.select_dtypes(include=[np.number]).dropna() 
        
        st.write(f"**Data Profile:** Detected {len(numeric_df.columns)} numeric features.")
        st.dataframe(numeric_df.head(5))
        
        if st.button("🧠 1-Click: Analyze & Train Model", type="primary"):
            with st.spinner("Building and training custom neural architecture..."):
                gen, data_scaler, latent_dim = train_auto_model(numeric_df, epochs=40)
                
                # Save dynamically generated model to session state
                st.session_state['auto_gen'] = gen
                st.session_state['auto_scaler'] = data_scaler
                st.session_state['auto_cols'] = numeric_df.columns
                st.session_state['latent_dim'] = latent_dim
                
    st.divider()
    
    # If a custom model is stored in session state, show the generation UI
    if 'auto_gen' in st.session_state:
        st.success("Custom Model Active in Memory!")
        st.subheader("Generate Synthetic Records")
        
        custom_num = st.number_input("Amount of rows to generate:", 10, 10000, 1000)
        
        if st.button("🧬 Generate Custom Synthetic Data"):
            gen = st.session_state['auto_gen']
            data_scaler = st.session_state['auto_scaler']
            cols = st.session_state['auto_cols']
            latent_dim = st.session_state['latent_dim']
            
            # 1. Generate Raw Latent Vectors
            z = torch.randn(custom_num, latent_dim)
            
            # 2. Pass through Custom Generator
            with torch.no_grad():
                synth_scaled = gen(z).numpy()
            
            # 3. Unscale targets back to real-world units automatically
            synth_real = data_scaler.inverse_transform(synth_scaled)
            
            # 4. Format DataFrame
            custom_synth_df = pd.DataFrame(synth_real, columns=cols)
            
            st.balloons()
            st.write(f"**Generated {custom_num} new synthetic rows matching your dataset's schema.**")
            st.dataframe(custom_synth_df.head(20))
            
            # Create a downloadable CSV button
            csv = custom_synth_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Synthetic CSV",
                data=csv,
                file_name='pumis_synthetic_clone.csv',
                mime='text/csv',
            )
            
            # Visual check for the first 2 columns
            if len(cols) >= 2:
                fig2, ax2 = plt.subplots(1, 2, figsize=(12, 4))
                sns.histplot(custom_synth_df[cols[0]], kde=True, ax=ax2[0], color="#10b981")
                ax2[0].set_title(cols[0])
                sns.scatterplot(data=custom_synth_df, x=cols[0], y=cols[1], ax=ax2[1], color="#005088", alpha=0.5)
                ax2[1].set_title(f"Correlation: {cols[0]} vs {cols[1]}")
                st.pyplot(fig2)