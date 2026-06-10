import streamlit as st
from tempfile import TemporaryDirectory
import os
from sip_systemsinsightpipeline.cld import Extract
from sip_systemsinsightpipeline.sdm import SDM
from sip_systemsinsightpipeline.plots import plot_simulated_intervention_ranking
import sys
import io
st.title('Diagrams-to-Dynamics (D2D): Exploring Causal Loop Diagram Leverage Points under Uncertainty')

# Upload file
uploaded_kumu_excel = st.file_uploader("Upload an Excel file (xlsx)", type="xlsx")

# Input fields (values stored but simulation doesn't run immediately)
# Initialize session state variables if they don't exist
if "N" not in st.session_state:
    st.session_state.N = "100"
if "time_unit" not in st.session_state:
    st.session_state.time_unit = "Months"
if "t_end" not in st.session_state:
    st.session_state.t_end = "12"
if "parameter_value_aux" not in st.session_state:
    st.session_state.parameter_value_aux = "0.3"
if "parameter_value_stocks" not in st.session_state:
    st.session_state.parameter_value_stocks = "0.1" 
if "seed" not in st.session_state:
    st.session_state.seed = "1912884"
if "cut_off_SA_importance" not in st.session_state:
    st.session_state.cut_off_SA_importance = "0.1"
if "double_factor_interventions_setting" not in st.session_state:
    st.session_state.double_factor_interventions_setting = "0"

# User inputs linked to session state
N = st.text_input("Enter the number of simulations to run (default 100)", st.session_state.N)
time_unit = st.text_input("Enter the base unit of time", st.session_state.time_unit)
t_end = st.text_input("Enter the final simulation time point", st.session_state.t_end)
parameter_value_aux = st.text_input("Enter the max parameter value for auxiliaries", st.session_state.parameter_value_aux)
parameter_value_stocks = st.text_input("Enter the max parameter value for stocks", st.session_state.parameter_value_stocks)
seed = st.text_input("Enter a seed for reproducibility (leave blank for random)", st.session_state.seed)
cut_off_SA_importance = st.text_input("Enter a cut-off for sensitivity coefficients to print (default rho>=0.1)", st.session_state.cut_off_SA_importance)
double_factor_interventions_setting = st.text_input("If you have interaction terms, do you want to simulate interventions on two factors simultaneously: 0=no, 1=yes (default: 0)?", st.session_state.double_factor_interventions_setting)

# Update session state when user changes input
if N != st.session_state.N:
    st.session_state.N = N
if time_unit != st.session_state.time_unit:
    st.session_state.time_unit = time_unit
if t_end != st.session_state.t_end:
    st.session_state.t_end = t_end
if parameter_value_aux != st.session_state.parameter_value_aux:
    st.session_state.parameter_value_aux = parameter_value_aux
if parameter_value_stocks != st.session_state.parameter_value_stocks:
    st.session_state.parameter_value_stocks = parameter_value_stocks
if seed != st.session_state.seed:
    st.session_state.seed = seed
if cut_off_SA_importance != st.session_state.cut_off_SA_importance:
    st.session_state.cut_off_SA_importance = cut_off_SA_importance
if double_factor_interventions_setting != st.session_state.double_factor_interventions_setting:
    st.session_state.double_factor_interventions_setting = double_factor_interventions_setting

# Button to confirm and run the simulation
if st.button("Run Simulation") and uploaded_kumu_excel is not None:
    with TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, uploaded_kumu_excel.name)

        with open(file_path, "wb") as f:
            f.write(uploaded_kumu_excel.getvalue())

        double_factor_interventions_setting_ = int(double_factor_interventions_setting) # Convert to boolean

        # Process files
        extract = Extract(file_path)
        s = extract.extract_settings(double_factor_interventions_setting_)

        # Convert inputs
        s.N = int(N)
        s.t_end = int(t_end)
        s.time_unit = time_unit
        s.parameter_value_aux = float(parameter_value_aux)
        s.parameter_value_stocks = float(parameter_value_stocks)
        s.prior = "uniform"
        s.seed = int(seed) if seed.strip() else None

        sdm = SDM(s)
        st.session_state.s = s
        st.session_state.sdm = sdm

        # Run simulations
        st.subheader("Simulated Intervention Rankings")

        df_sol, param_samples = sdm.run_simulations()

        intervention_effects_per_voi = sdm.get_intervention_effects()

        st.session_state.df_sol = df_sol
        st.session_state.param_samples = param_samples
        st.session_state.intervention_effects = intervention_effects_per_voi

        # Display results
        for voi in s.variable_of_interest:
            fig_var_rank = plot_simulated_intervention_ranking(s, intervention_effects_per_voi[voi], voi)
            st.pyplot(fig_var_rank)

    # Sensitivity Analysis
    int_var = None
    st.subheader("Sensitivity Analysis Results")

    for voi in s.variable_of_interest:
        st.write(f"**Variable of Interest: {voi}**")  # Display VOI in bold
        # Capture print output
        output_buffer = io.StringIO()
        sys.stdout = output_buffer  # Redirect print statements to buffer
        SA_results, df_SA = sdm.run_SA(voi, int_var, float(cut_off_SA_importance))
        sys.stdout = sys.__stdout__  # Reset stdout to normal
    
        # Display captured output in Streamlit
        st.text(output_buffer.getvalue()) 

