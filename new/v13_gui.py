import streamlit as st
import pandas as pd
import os
import sys
import io
from pathlib import Path
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Add 'model' folder to sys.path to import vADVANCED13.py
model_path = Path(__file__).parent / "model"
sys.path.append(str(model_path))

# Attempt to import vADVANCED13
try:
    import vADVANCED13
except ImportError:
    st.error(
        "❌ Could not find `vADVANCED13.py` in the `model` folder. Please ensure the directory structure is correct."
    )
    st.info("Current Directory: " + os.getcwd())
    st.stop()

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="ProToVec | Advanced AI Industrial Scheduler",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================
# CUSTOM STYLING (Premium Look)
# ==========================================
st.markdown(
    """
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background-color: #f8f9fa;
    }
    
    /* Header */
    .stHeader {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 10px;
        background: #f1f4f9;
        padding: 0.8rem 1.5rem;
        border-radius: 6px;
        color: #1a335e;
        margin-bottom: 1rem;
        border-bottom: 2px solid #1e3c72;
    }
    
    .main-title {
        font-size: 1.2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.2px;
        color: #1a335e;
    }
    
    .sub-title {
        font-size: 0.85rem;
        opacity: 0.8;
        font-weight: 400;
        color: #444;
    }

    /* Cards */
    .stMetric {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border: 1px solid #ddd;
        color: #1a335e !important;
    }
    
    .stMetric label, .stMetric div {
        color: #1a335e !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #fff;
        border-radius: 8px 8px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        border: 1px solid #eee;
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        background-color: #1e3c72 !important;
        color: white !important;
        border: none !important;
    }

    /* Buttons */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        padding: 0.6rem 2rem;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }

    /* Table/Data Editor Alignment */
    /* Target all div and span elements within the dataframe container to force center alignment */
    [data-testid="stDataFrame"] div, [data-testid="stDataFrame"] span {
        text-align: center !important;
        justify-content: center !important;
    }
    
    /* Ensure the data editor also adheres to center alignment */
    div[data-testid="stDataEditor"] div, div[data-testid="stDataEditor"] span {
        text-align: center !important;
        justify-content: center !important;
    }
    
    /* Center align the grid cells themselves */
    .glideDataEditor-container div {
        text-align: center !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================
if "wagon_df" not in st.session_state:
    # Try to load existing config
    wagon_csv = model_path / "wagon_config.csv"
    if wagon_csv.exists():
        st.session_state.wagon_df = pd.read_csv(wagon_csv)
    else:
        st.session_state.wagon_df = pd.DataFrame(
            columns=[
                "Transporter Name",
                "Superfast Speed",
                "Fast Speed",
                "Slow Speed",
                "Lift Time",
                "Lower Time",
                "Minimum Station No",
                "Maximum Station No",
                "Basic Position",
                "Wagon Length (mm)",
                "Safety Buffer (mm)",
                "Size Factor",
                "Station Occupy Factor",
                "Row",
            ]
        )

if "tanks_df" not in st.session_state:
    tanks_csv = model_path / "tanks_csv.csv"
    if tanks_csv.exists():
        st.session_state.tanks_df = pd.read_csv(tanks_csv)
    else:
        st.session_state.tanks_df = pd.DataFrame(
            columns=[
                "project_id",
                "station_no",
                "process_name",
                "critical_status",
                "distance_mm",
                "dip_time_sec",
                "Row",
            ]
        )

if "results" not in st.session_state:
    st.session_state.results = None

# ==========================================
# HEADER SECTION
# ==========================================
with st.container():
    st.markdown(
        """
    <div class="stHeader">
        <h1 class="main-title">🏭 Protovec Advanced Scheduler</h1>
        <p class="sub-title">Powered by vADVANCED13 Physics & AI Engine</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

# Sidebar removed as requested
enable_collision = True
safe_dist = 0.0

# ==========================================
# MAIN INTERFACE - TABS
# ==========================================
tab1, tab2, tab3 = st.tabs(["🏗️ Configuration", "📊 Execution", "🗂️ Analysis"])

# --- TAB 1: CONFIGURATION & EXECUTION ---
with tab1:
    # --- 1. Container for Run Engine (MOVED TO TOP UI but filled LATER) ---
    run_engine_container = st.container()

    st.divider()

    # --- 2. Wagon Configuration Section ---
    st.markdown("### 🚂 Step 2: Wagon Settings")
    edited_wagon_df = st.data_editor(
        st.session_state.wagon_df,
        num_rows="dynamic",
        use_container_width=True,
        height=200,  # Smaller fixed height
        key="wagon_editor_stacked",
        column_config={
            "Minimum Station No": st.column_config.NumberColumn(step=1),
            "Maximum Station No": st.column_config.NumberColumn(step=1),
            "Row": st.column_config.NumberColumn(step=1),
        },
    )

    # Unified Action Row for Wagons
    ca1, ca2 = st.columns(2)
    with ca1:
        wagon_file = st.file_uploader(
            "Upload Wagon CSV",
            type=["csv"],
            key="up_wagon",
            label_visibility="collapsed",
        )
        if wagon_file:
            st.session_state.wagon_df = pd.read_csv(wagon_file)
            st.rerun()
    with ca2:
        if st.button("💾 SAVE WAGON CONFIG", use_container_width=True):
            edited_wagon_df.to_csv(model_path / "wagon_config.csv", index=False)
            st.success("Wagons Saved!")

    st.divider()

    # --- 3. Station Data Section ---
    st.markdown("### 📍 Step 3: Station Data")
    edited_tanks_df = st.data_editor(
        st.session_state.tanks_df,
        num_rows="dynamic",
        use_container_width=True,
        height=250,  # Optimized height
        key="tanks_editor_stacked",
        column_config={
            "station_no": st.column_config.NumberColumn(step=1),
            "Row": st.column_config.NumberColumn(step=1),
        },
    )

    # Unified Action Row for Tanks
    cb1, cb2 = st.columns(2)
    with cb1:
        tanks_file = st.file_uploader(
            "Upload Station CSV",
            type=["csv"],
            key="up_tanks",
            label_visibility="collapsed",
        )
        if tanks_file:
            st.session_state.tanks_df = pd.read_csv(tanks_file)
            st.rerun()
    with cb2:
        if st.button("💾 SAVE STATION DATA", use_container_width=True):
            edited_tanks_df.to_csv(model_path / "tanks_csv.csv", index=False)
            st.success("Stations Saved!")

    # --- 4. Load Balancing Preview ---
    with st.expander("📊 Analyze Parallel Station Groups (Load Balancing)"):
        summary_data = []
        if not edited_tanks_df.empty:
            # Group consecutive stations with same process name (same logic as vADVANCED13)
            temp_tanks = edited_tanks_df.to_dict("records")
            if temp_tanks:
                curr_group = [temp_tanks[0]]
                for i in range(1, len(temp_tanks)):
                    if temp_tanks[i].get("process_name") == curr_group[-1].get(
                        "process_name"
                    ) and temp_tanks[i].get("process_name"):
                        curr_group.append(temp_tanks[i])
                    else:
                        summary_data.append(
                            {
                                "Process": curr_group[0].get("process_name"),
                                "Stations": ", ".join(
                                    [str(t.get("station_no")) for t in curr_group]
                                ),
                                "Count": len(curr_group),
                                "Logic": "Load Balancing"
                                if len(curr_group) > 1
                                else "Single Tank",
                            }
                        )
                        curr_group = [temp_tanks[i]]
                summary_data.append(
                    {
                        "Process": curr_group[0].get("process_name"),
                        "Stations": ", ".join(
                            [str(t.get("station_no")) for t in curr_group]
                        ),
                        "Count": len(curr_group),
                        "Logic": "Load Balancing"
                        if len(curr_group) > 1
                        else "Single Tank",
                    }
                )
            st.table(pd.DataFrame(summary_data))
            st.caption(
                "When multiple stations have the same process name consecutively, the engine distributes loads across them to increase throughput."
            )

    # --- POPULATE STEP 1 (RUN ENGINE) AT THE TOP ---
    with run_engine_container:
        st.markdown("### 🚀 Step 1: Run Engine")
        c_run1, c_run2, c_run3, c_run4 = st.columns([1, 1, 1, 1])
        with c_run1:
            total_loads = st.number_input(
                "Number of Loads (Concurrent)",
                min_value=1,
                value=1,
                step=1,
                help="Simulate multiple loads moving through the plant at the same time. The engine will interleave wagon tasks and distribute loads across parallel stations (Load Balancing).",
            )
            if total_loads > 1:
                st.info(
                    f"💡 {total_loads} loads will be interleaved. Parallel tanks for the same process will be balanced."
                )
        with c_run2:
            processing_mode = st.selectbox(
                "Processing Mode",
                options=["grouped", "sequential", "load_balance"],
                index=0,
                help="grouped: Multi-source to multi-destination routing (your custom flow)\nsequential: Process tanks in sequence with criticality\nload_balance: Original load rotation logic",
            )
            if processing_mode == "grouped":
                st.info("📦 Grouped: Items flow between process groups")
            elif processing_mode == "sequential":
                st.info("🔢 Sequential: Tank 1 → 2 → 3 with criticality")
            else:
                st.info("⚖️ Load Balance: Distributes across parallel tanks")
        with c_run3:
            st.write("")  # Spacer
            run_button = st.button(
                "▶️ START PHYSICS CALCULATION", type="primary", use_container_width=True
            )
        with c_run4:
            st.write("")  # Spacer
            st.caption(f"Mode: {processing_mode}")

        if run_button:
            with st.spinner("Analyzing physics..."):
                try:
                    tanks_list = edited_tanks_df.to_dict("records")
                    raw_config = edited_wagon_df.to_dict("records")
                    wagon_config = {}
                    for row in raw_config:
                        name = row["Transporter Name"]
                        # CRITICAL FIX: Use 'Row' to match tank data logic in vADVANCED13
                        w_row = row.get("Row") or row.get("Row Number") or 1
                        wagon_config[name] = {
                            "sf": float(row["Superfast Speed"]),
                            "f": float(row["Fast Speed"]),
                            "s": float(row["Slow Speed"]),
                            "lift_time": float(row["Lift Time"]),
                            "lower_time": float(row["Lower Time"]),
                            "row": int(w_row),
                            "min_stn": int(row["Minimum Station No"]),
                            "max_stn": int(row["Maximum Station No"]),
                            "basic_pos": int(row.get("Basic Position", 0)),
                            "length": float(row.get("Wagon Length (mm)", 0)),
                            "buffer": float(row.get("Safety Buffer (mm)", 0)),
                            "size_factor": float(row.get("Size Factor", 1.0)),
                            "occupy_factor": float(
                                row.get("Station Occupy Factor", 1.0)
                            ),
                        }

                    sequence_data = vADVANCED13.generate_sequence_from_data(
                        tanks_data=tanks_list,
                        config=wagon_config,
                        num_loads=total_loads,
                        processing_mode=processing_mode,
                        verbose=False,
                    )

                    if sequence_data and len(sequence_data) > 1:
                        headers = sequence_data[0]
                        rows = sequence_data[1:]
                        st.session_state.results = pd.DataFrame(rows, columns=headers)
                        st.success(
                            f"✅ Done! Generated {len(rows)} steps in '{processing_mode}' mode. See 'Execution' tab."
                        )
                    else:
                        st.error("Empty sequence. Check Wagon Rows vs Station Rows.")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# --- TAB 2: EXECUTION (RESULTS) ---
with tab2:
    if st.session_state.results is None:
        st.info(
            "No sequence generated yet. Please configure and run the engine in the 'Configuration' tab."
        )
    else:
        st.subheader("� Execution Plan")

        # Color coding function
        def highlight_collision(row):
            if "CollisionFlag" in row.index:
                if row["CollisionFlag"] == "COLLISION":
                    return ["background-color: #ffcccc"] * len(row)
                elif row["CollisionFlag"] == "PROXIMITY":
                    return ["background-color: #fff4cc"] * len(row)
            return [""] * len(row)

        res_df = st.session_state.results
        st.dataframe(
            res_df.style.apply(highlight_collision, axis=1),
            use_container_width=True,
            height=500,
        )

        # Download results
        csv = res_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Sequence (CSV)",
            data=csv,
            file_name=f"sequence_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
        )

# --- TAB 3: ANALYSIS ---
with tab3:
    if st.session_state.results is None:
        st.warning("Please generate a sequence in the 'Execution' tab first.")
    else:
        df = st.session_state.results.copy()

        # KPI Row
        kp1, kp2, kp3, kp4 = st.columns(4)
        with kp1:
            total_time = pd.to_numeric(df["AccumulatedTime"]).max()
            st.metric("Total Processing Cycle", f"{total_time:.1f}s")
        with kp2:
            avg_wait = (
                pd.to_numeric(df["WaitTime"]).mean() if "WaitTime" in df.columns else 0
            )
            st.metric("Avg Wait/Interlock", f"{avg_wait:.1f}s")
        with kp3:
            # Count distinct wait actions
            waits = len(df[df["Command"].str.contains("WAIT", na=False)])
            st.metric("Anti-Collision Actions", waits, delta_color="inverse")
        with kp4:
            steps = len(df)
            st.metric("Command Density", f"{steps} cmds")

        st.divider()

        # NEW: Dip Time Analytics Section
        st.subheader("🧪 Actual Dip Time Analytics")
        try:
            # Prepare data for calculation
            tanks_list = st.session_state.tanks_df.to_dict("records")
            seq_list = [
                st.session_state.results.columns.tolist()
            ] + st.session_state.results.values.tolist()

            dip_data = vADVANCED13.calculate_actual_dip_times(tanks_list, seq_list)

            if dip_data:
                dip_df = pd.DataFrame(dip_data)

                # Show key metrics for dips
                d1, d2, d3 = st.columns(3)
                with d1:
                    avg_variance = pd.to_numeric(
                        dip_df["Variance"].str.replace("s", "")
                    ).mean()
                    st.metric("Avg Dip Variance", f"{avg_variance:.2f}s")
                with d2:
                    total_dips = len(dip_df)
                    st.metric("Total Dips Tracked", total_dips)
                with d3:
                    max_err = pd.to_numeric(
                        dip_df["Variance"].str.replace("s", "")
                    ).max()
                    st.metric("Max Dip Delay", f"{max_err:.2f}s")

                st.dataframe(dip_df, use_container_width=True)

                # Visual comparison
                dip_df["FloatVar"] = pd.to_numeric(
                    dip_df["Variance"].str.replace("s", "")
                )
                fig_dip = px.bar(
                    dip_df,
                    x="Station",
                    y="FloatVar",
                    color="Process",
                    title="Dip Time Variance per Station (Actual vs Target)",
                    labels={"FloatVar": "Variance (s)", "Station": "Station No"},
                )
                st.plotly_chart(fig_dip, use_container_width=True)
            else:
                st.info("No dip events detected in the current sequence.")
        except Exception as dip_err:
            st.error(f"Dip Analytics Error: {dip_err}")

        st.divider()

        # TIMELINE CHART (Gantt style using px.bar for stability with numbers)
        st.subheader("⏱️ Multi-Wagon Synchronization Timeline")

        try:
            # Prepare data
            df["AccumulatedTime"] = pd.to_numeric(df["AccumulatedTime"])
            df["TravelTime"] = pd.to_numeric(df["TravelTime"])

            # Use px.bar as it handles numeric x-axis better than px.timeline for custom units
            fig = px.bar(
                df,
                base="AccumulatedTime",
                x="TravelTime",
                y="Wagon",
                color="Command",
                orientation="h",
                hover_data=["Step No", "Value", "CollisionFlag"],
                title="Wagon Activity Over Time (Seconds)",
                color_discrete_sequence=px.colors.qualitative.Prism,
            )

            fig.update_layout(xaxis_title="Time (seconds)", yaxis_title="Wagon ID")
            st.plotly_chart(fig, use_container_width=True)

            # STEP CHART
            st.subheader("📈 Station Sequence Path")
            fig2 = px.line(
                df,
                x="AccumulatedTime",
                y="Value",
                color="Wagon",
                markers=True,
                labels={"Value": "Station No", "AccumulatedTime": "Time (s)"},
                title="Wagon Movement Path",
            )
            st.plotly_chart(fig2, use_container_width=True)

        except Exception as vis_err:
            st.error(f"Visualization Error: {vis_err}")

# ==========================================
# FOOTER
# ==========================================
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "ProToVec v13.0 | Advanced Industrial Automation Tool | © 2026"
    "</div>",
    unsafe_allow_html=True,
)
