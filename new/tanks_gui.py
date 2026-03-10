import streamlit as st
import pandas as pd
import os
import io
import sys
from pathlib import Path

# Add the code directory to path to import vADVANCED13
sys.path.insert(0, str(Path(__file__).parent))
import vADVANCED13

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Tank Sequence Generator",
    layout="wide",
    page_icon="🚂"
)

# ==========================================
# STYLE
# ==========================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        color: #666;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# HEADER
# ==========================================
st.markdown('<div class="main-header">🚂 Tank Sequence Generator</div>', unsafe_allow_html=True)
st.caption("Using vADVANCED13 Model - Generate sequences from tank data")

# ==========================================
# INITIALIZE SESSION STATE
# ==========================================
if "df_tanks" not in st.session_state:
    # Load default data from CSV if exists
    csv_path = os.path.join(os.path.dirname(__file__), "tanks_csv.csv")
    if os.path.exists(csv_path):
        st.session_state.df_tanks = pd.read_csv(csv_path)
    else:
        # Default empty structure
        st.session_state.df_tanks = pd.DataFrame({
            "project_id": [2],
            "station_no": [1],
            "process_name": [""],
            "critical_status": ["Low"],
            "distance_mm": [0],
            "dip_time_sec": [0]
        })

if "generated_sequence" not in st.session_state:
    st.session_state.generated_sequence = None

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def generate_sequence_wrapper(tanks_df):
    """
    Wrapper to generate sequence using vADVANCED13 with direct data input.
    This avoids CSV file collisions by passing data directly.
    
    Args:
        tanks_df: pandas DataFrame with tank data
    
    Returns:
        pandas DataFrame with generated sequence
    """
    try:
        # Load wagon config
        config = vADVANCED13.load_wagon_config()
        if not config:
            raise Exception("No wagon configuration found. Please ensure wagon_config.csv exists.")
        
        if len(tanks_df) == 0:
            raise Exception("No tank data provided.")
        
        # Generate sequence using direct data input (no CSV file needed)
        sequence_data = vADVANCED13.generate_sequence_from_data(
            tanks_data=tanks_df,
            config=config,
            verbose=False  # Don't print in GUI mode
        )
        
        if sequence_data is None or len(sequence_data) <= 1:
            raise Exception("Sequence generation completed but no moves were generated.")
        
        # Convert to DataFrame
        headers = sequence_data[0]
        rows = sequence_data[1:]
        df = pd.DataFrame(rows, columns=headers)
        return df
        
    except Exception as e:
        raise Exception(f"Error in sequence generation: {str(e)}")

# ==========================================
# TABS
# ==========================================
tab_input, tab_generated = st.tabs([
    "📝 Tank Data Input",
    "📊 Generated Sequence"
])

# ==========================================
# TAB 1: TANK DATA INPUT
# ==========================================
with tab_input:
    st.subheader("Edit Tank/Station Data")
    st.markdown("Enter or modify the tank data below. This replaces the CSV file input.")
    
    # Data editor
    edited_df = st.data_editor(
        st.session_state.df_tanks,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "project_id": st.column_config.NumberColumn("Project ID", min_value=1),
            "station_no": st.column_config.NumberColumn("Station No", min_value=1),
            "process_name": st.column_config.TextColumn("Process Name"),
            "critical_status": st.column_config.SelectboxColumn(
                "Critical Status",
                options=["High", "Low"]
            ),
            "distance_mm": st.column_config.NumberColumn("Distance (mm)", min_value=0),
            "dip_time_sec": st.column_config.NumberColumn("Dip Time (sec)", min_value=0)
        }
    )
    
    st.session_state.df_tanks = edited_df
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("🚀 Generate Sequence", type="primary", use_container_width=True):
            if len(edited_df) == 0:
                st.error("Please add at least one row of tank data.")
            else:
                with st.spinner("Generating sequence using vADVANCED13 model..."):
                    try:
                        # Generate sequence directly from DataFrame (no CSV file needed)
                        sequence_df = generate_sequence_wrapper(edited_df)
                        
                        if sequence_df is not None and len(sequence_df) > 0:
                            st.session_state.generated_sequence = sequence_df
                            st.success(f"✅ Sequence generated successfully! ({len(sequence_df)} steps)")
                            st.info("Switch to the 'Generated Sequence' tab to view results.")
                        else:
                            st.warning("⚠️ Sequence generation completed but no data was returned.")
                            
                    except Exception as e:
                        st.error(f"❌ Error generating sequence: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
    
    with col2:
        if st.button("💾 Save to CSV", use_container_width=True):
            csv_path = os.path.join(os.path.dirname(__file__), "tanks_csv.csv")
            edited_df.to_csv(csv_path, index=False)
            st.success(f"✅ Saved to {csv_path}")
    
    with col3:
        # File uploader
        uploaded_file = st.file_uploader("📁 Upload CSV", type=['csv'], help="Upload a CSV file to load tank data")
        if uploaded_file is not None:
            try:
                df_uploaded = pd.read_csv(uploaded_file)
                # Validate columns
                required_cols = ["project_id", "station_no", "process_name", "critical_status", "distance_mm", "dip_time_sec"]
                if all(col in df_uploaded.columns for col in required_cols):
                    st.session_state.df_tanks = df_uploaded
                    st.success("✅ CSV loaded successfully!")
                    st.rerun()
                else:
                    st.error(f"❌ CSV must contain these columns: {', '.join(required_cols)}")
            except Exception as e:
                st.error(f"❌ Error loading CSV: {str(e)}")

# ==========================================
# TAB 2: GENERATED SEQUENCE
# ==========================================
with tab_generated:
    st.subheader("📊 Generated Sequence")
    
    # Check if sequence exists
    if st.session_state.generated_sequence is None:
        # Show upload option when no sequence exists
        st.markdown("### 📁 Load Sequence")
        col_upload, col_info = st.columns([1, 2])
        with col_upload:
            uploaded_sequence = st.file_uploader(
                "📁 Upload Generated Sequence CSV", 
                type=['csv'], 
                help="Upload a previously generated sequence CSV to view it",
                key="upload_sequence"
            )
            if uploaded_sequence is not None:
                try:
                    df_uploaded_seq = pd.read_csv(uploaded_sequence)
                    # Check if it looks like a generated sequence (has Wagon, Command, etc.)
                    if "Wagon" in df_uploaded_seq.columns and "Command" in df_uploaded_seq.columns:
                        st.session_state.generated_sequence = df_uploaded_seq
                        st.success("✅ Generated sequence loaded successfully!")
                        st.rerun()
                    else:
                        st.warning("⚠️ This doesn't look like a generated sequence file. Expected columns: Wagon, Command, Step No, etc.")
                except Exception as e:
                    st.error(f"❌ Error loading sequence: {str(e)}")
        
        with col_info:
            st.info("👈 **To generate a sequence:**\n1. Go to 'Tank Data Input' tab\n2. Add/edit your stations\n3. Click '🚀 Generate Sequence' button\n4. Come back here to view the results!")
    
    # Display generated sequence if it exists
    if st.session_state.generated_sequence is not None:
        df_seq = st.session_state.generated_sequence.copy()
        
        # Header with clear button
        col_header, col_clear = st.columns([3, 1])
        with col_header:
            st.success(f"✅ **Sequence Generated!** Total Steps: {len(df_seq)}")
        with col_clear:
            if st.button("🔄 Clear Sequence", use_container_width=True):
                st.session_state.generated_sequence = None
                st.rerun()
        
        st.markdown("---")
        st.markdown("### 📋 Sequence Table")
        
        # Display sequence table
        st.dataframe(
            df_seq,
            use_container_width=True,
            height=400,
            hide_index=True
        )
        
        st.markdown("---")
        
        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            unique_wagons = df_seq["Wagon"].nunique() if "Wagon" in df_seq.columns else 0
            st.metric("Wagons Used", unique_wagons)
        
        with col2:
            total_steps = len(df_seq)
            st.metric("Total Steps", total_steps)
        
        with col3:
            if "AccumulatedTime" in df_seq.columns:
                try:
                    max_time = pd.to_numeric(df_seq["AccumulatedTime"], errors='coerce').max()
                    st.metric("Max Time (sec)", f"{max_time:.2f}" if pd.notna(max_time) else "N/A")
                except:
                    st.metric("Max Time (sec)", "N/A")
            else:
                st.metric("Max Time (sec)", "N/A")
        
        with col4:
            if "Command" in df_seq.columns:
                get_from_count = len(df_seq[df_seq["Command"] == "GET FROM"])
                st.metric("GET FROM Commands", get_from_count)
            else:
                st.metric("GET FROM Commands", "N/A")
        
        st.markdown("---")
        
        # Visualizations
        if "Wagon" in df_seq.columns and "AccumulatedTime" in df_seq.columns:
            st.subheader("📊 Sequence Timeline by Wagon")
            
            # Prepare data for timeline
            try:
                df_viz = df_seq.copy()
                df_viz["AccumulatedTime"] = pd.to_numeric(df_viz["AccumulatedTime"], errors='coerce')
                df_viz["TravelTime"] = pd.to_numeric(df_viz["TravelTime"], errors='coerce')
                
                # Create timeline data
                timeline_data = []
                for idx, row in df_viz.iterrows():
                    start_time = row["AccumulatedTime"] if pd.notna(row["AccumulatedTime"]) else 0
                    duration = row["TravelTime"] if pd.notna(row["TravelTime"]) else 0
                    
                    timeline_data.append({
                        "Wagon": row["Wagon"],
                        "Command": row.get("Command", "Unknown"),
                        "Start": start_time,
                        "End": start_time + duration,
                        "Step": row.get("Step No", idx + 1)
                    })
                
                df_timeline = pd.DataFrame(timeline_data)
                
                # Simple bar chart
                if len(df_timeline) > 0:
                    st.bar_chart(
                        df_timeline.set_index("Wagon")[["Start", "End"]],
                        height=300
                    )
            except Exception as e:
                st.warning(f"Could not generate timeline visualization: {str(e)}")
        
        # Download button
        st.markdown("---")
        csv_buffer = io.StringIO()
        df_seq.to_csv(csv_buffer, index=False)
        
        st.download_button(
            "⬇️ Download Generated Sequence as CSV",
            data=csv_buffer.getvalue(),
            file_name="generated_sequence.csv",
            mime="text/csv",
            type="primary"
        )

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("## ⚙️ Statistics")
    
    df_current = st.session_state.df_tanks
    
    st.metric("Total Tanks/Stations", len(df_current))
    
    if "dip_time_sec" in df_current.columns:
        try:
            total_dip = pd.to_numeric(df_current["dip_time_sec"], errors='coerce').sum()
            st.metric("Total Dip Time (sec)", f"{total_dip:.1f}" if pd.notna(total_dip) else "0")
        except:
            st.metric("Total Dip Time (sec)", "0")
    
    if "distance_mm" in df_current.columns:
        try:
            max_dist = pd.to_numeric(df_current["distance_mm"], errors='coerce').max()
            st.metric("Max Distance (mm)", f"{max_dist:.0f}" if pd.notna(max_dist) else "0")
        except:
            st.metric("Max Distance (mm)", "0")
    
    st.markdown("---")
    st.markdown("### 📖 Instructions")
    st.markdown("""
    1. **Input Tab**: Add or edit tank/station data
    2. Click **Generate Sequence** to run vADVANCED13 model
    3. **Generated Sequence Tab**: View the output sequence
    4. Download the sequence as CSV if needed
    """)
    
    st.markdown("---")
    st.markdown("### 🔧 Model Info")
    st.caption("Using: vADVANCED13.py")
    st.caption("Features: Transformer + GNN + PPO")
