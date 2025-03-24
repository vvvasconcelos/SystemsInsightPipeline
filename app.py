import streamlit as st
from tempfile import TemporaryDirectory
import os
from systemdynamics.cld import Extract
from systemdynamics.sdm import SDM
from systemdynamics.plots import plot_simulated_intervention_ranking
import sys
import io
st.title('Diagrams to Dynamics: A System Dynamics Analysis of a Causal Loop Diagram')

# Upload file
uploaded_kumu_excel = st.file_uploader("Upload an Excel file (xlsx)", type="xlsx")

# Input fields (values stored but simulation doesn't run immediately)
# N = st.text_input("Enter the number of simulations to run (default 100)", "100")
# time_unit = st.text_input("Enter the base unit of time (default: Months)", "Months")
# t_end = st.text_input("Enter the final simulation time point in specified time units (default: 12)", "12")
# parameter_value = st.text_input("Enter the maximum parameter value theta (default 0.5)", "0.5")
# Initialize session state variables if they don't exist
if "N" not in st.session_state:
    st.session_state.N = "100"
if "time_unit" not in st.session_state:
    st.session_state.time_unit = "Months"
if "t_end" not in st.session_state:
    st.session_state.t_end = "12"
if "parameter_value" not in st.session_state:
    st.session_state.parameter_value = "0.5"
if "seed" not in st.session_state:
    st.session_state.seed = "1912884"

# User inputs linked to session state
N = st.text_input("Enter the number of simulations to run (default 100)", st.session_state.N)
time_unit = st.text_input("Enter the base unit of time (default: Months)", st.session_state.time_unit)
t_end = st.text_input("Enter the final simulation time point (default: 12)", st.session_state.t_end)
parameter_value = st.text_input("Enter the max parameter value theta (default 0.5)", st.session_state.parameter_value)
seed = st.text_input("Enter a seed for reproducibility (leave blank for random)", st.session_state.seed)

# Update session state when user changes input
if N != st.session_state.N:
    st.session_state.N = N
if time_unit != st.session_state.time_unit:
    st.session_state.time_unit = time_unit
if t_end != st.session_state.t_end:
    st.session_state.t_end = t_end
if parameter_value != st.session_state.parameter_value:
    st.session_state.parameter_value = parameter_value
# Update seed in session state
if seed != st.session_state.seed:
    st.session_state.seed = seed

# Button to confirm and run the simulation
if st.button("Run Simulation") and uploaded_kumu_excel is not None:
    with TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, uploaded_kumu_excel.name)

        with open(file_path, "wb") as f:
            f.write(uploaded_kumu_excel.getvalue())

        # Process files
        extract = Extract(file_path)
        s = extract.extract_settings()

        # Convert inputs
        s.N = int(N)
        s.t_end = int(t_end)
        s.time_unit = time_unit
        s.parameter_value = float(parameter_value)
        s.prior = "uniform"
        # Set seed (random if left blank)
        s.seed = int(seed) if seed.strip() else None


        sdm = SDM(s)
        st.session_state.s = s
        st.session_state.sdm = sdm

        # Run simulations
        st.subheader("Simulated Intervention Rankings")

        df_sol, param_samples, eig_val_vec = sdm.run_simulations()
        intervention_effects_per_voi = sdm.get_intervention_effects()

        st.session_state.df_sol = df_sol
        st.session_state.param_samples = param_samples
        st.session_state.intervention_effects = intervention_effects_per_voi

        # Display results
        for voi in s.variable_of_interest:
            fig_var_rank = plot_simulated_intervention_ranking(s, intervention_effects_per_voi[voi], voi)
            st.pyplot(fig_var_rank)

    # Sensitivity Analysis
    cut_off_SA_importance = 0.05
    int_var = None
    st.subheader("Sensitivity Analysis Results (>rho=0.05)")

    for voi in s.variable_of_interest:
        st.write(f"**Variable of Interest: {voi}**")  # Display VOI in bold
        # Capture print output
        output_buffer = io.StringIO()
        sys.stdout = output_buffer  # Redirect print statements to buffer

        SA_results, df_SA = sdm.run_SA(voi, int_var, cut_off_SA_importance)

        sys.stdout = sys.__stdout__  # Reset stdout to normal
    
        # Display captured output in Streamlit
        st.text(output_buffer.getvalue()) 

