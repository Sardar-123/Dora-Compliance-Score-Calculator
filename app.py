import streamlit as st
import pandas as pd
import joblib
import numpy as np
import openai
import time

# Load the trained model and label encoders
best_rf_model = joblib.load('dora_compliance_rf_model_final.pkl')
label_encoders = joblib.load('label_encoders_final.pkl')  # Load the label encoders

# Initialize the OpenAI client with the Azure API key
with open("C:/Users/mohds/Desktop/Project-2024/Dora_Compliance_Ml_Model/.azure_openai_key.txt") as f:
    key = f.read().strip()

# Initialize the OpenAI client with Azure settings
openai.api_type = "azure"
openai.api_base = "https://dora-llm-workspace.openai.azure.com/" 
openai.api_key = key  
openai.api_version = "2023-05-15"  

# Define testing types
testing_types = [
    'Source code review', 'Scenario-based testing', 'Compatibility testing', 'Performance testing',
    'End-to-end testing', 'Open-source analysis', 'Vulnerability assessments/scans',
    'Network security assessments', 'Physical security reviews', 'Penetration Testing',
    'Thread-Led Penetration Testing', 'BCP testing', 'Crisis comms plan testing', 'Backup & Restore',
    'DR Testing', 'IT Continuity Plan', 'Accessibility Testing'
]

# Define the feature columns used in training
feature_columns = [
    'frequency_of_assessment', 'application_code_stack', 'test_data_management', 'documentation', 
    'critical_business_process_covered', 'metrics_and_kpis', 'production_data_usage', 'test_automation'
]

# Initialize session state to store user inputs for each testing type
if 'user_inputs' not in st.session_state:
    st.session_state.user_inputs = []

# Streamlit UI
st.title("DORA Compliance Score Calculator")

# Function to render input fields for a specific testing type
def render_inputs(input_index):
    with st.form(key=f"form_{input_index}"):
        selected_testing_type = st.selectbox("Select Testing Type", testing_types, key=f"test_type_{input_index}")

        frequency_of_assessment = st.selectbox("Frequency of Assessment", ['Daily', 'Weekly', 'Monthly', 'Quarterly', 'Yearly'], key=f"freq_{input_index}")
        critical_business_process_covered = st.selectbox("Critical Business Process Covered", ['High', 'Moderate', 'Low'], key=f"cbp_{input_index}")
        application_code_stack = st.selectbox("Application Code Stack", ['Mainframe', 'Java', 'API'], key=f"app_stack_{input_index}")
        production_data_usage = st.selectbox("Production Data Usage", ['Yes', 'No'], key=f"prod_data_{input_index}")
        test_data_management = st.selectbox("Test Data Management", ['Manual', 'Automated', 'Hybrid'], key=f"test_data_{input_index}")
        test_automation = st.selectbox("Test Automation", ['Yes', 'No'], key=f"test_automation_{input_index}")
        metrics_and_kpis = st.selectbox("Metrics and KPIs", ['High', 'Moderate', 'Low'], key=f"metrics_{input_index}")
        documentation = st.selectbox("Documentation", ['Poor', 'Average', 'Good', 'Excellent'], key=f"doc_{input_index}")

        # Save button for this form
        submit_button = st.form_submit_button(label="Save Inputs for Selected Testing Type")

        if submit_button:
            # Check for missing values
            if any(x is None or x == "" for x in [selected_testing_type, frequency_of_assessment, critical_business_process_covered,
                                                   application_code_stack, production_data_usage, test_data_management,
                                                   test_automation, metrics_and_kpis, documentation]):
                st.error("Please fill in all fields for the selected testing type.")
            else:
                input_data = {
                    'testing_type': selected_testing_type,
                    'frequency_of_assessment': frequency_of_assessment,
                    'critical_business_process_covered': critical_business_process_covered,
                    'application_code_stack': application_code_stack,
                    'production_data_usage': production_data_usage,
                    'test_data_management': test_data_management,
                    'test_automation': test_automation,
                    'metrics_and_kpis': metrics_and_kpis,
                    'documentation': documentation
                }
                st.session_state.user_inputs.append(input_data)  # Add to session state only if valid
                st.success(f"Inputs for {selected_testing_type} saved!")

# Loop for adding multiple testing types
input_index = len(st.session_state.user_inputs)  # Set the index based on existing inputs
render_inputs(input_index)  # Render the current set of input fields

# Option to add another testing type
if st.button("Add Another Testing Type"):
    st.session_state.user_inputs.append({})  # Append an empty dict for the new testing type
    st.experimental_rerun()  # Rerun the app to show the new input fields

# Function to generate recommendations for DORA compliance improvements and identify missing testing types
def generate_compliance_recommendations(individual_scores, valid_inputs, threshold=80, delay=2):
    recommendations = []
    
    for i, score in enumerate(individual_scores):
        if score < threshold:
            # Extracting the necessary fields from the valid inputs
            input_data = valid_inputs[i]
            selected_testing_type = input_data['testing_type']
            frequency_of_assessment = input_data['frequency_of_assessment']
            critical_business_process_covered = input_data['critical_business_process_covered']
            application_code_stack = input_data['application_code_stack']
            production_data_usage = input_data['production_data_usage']
            test_data_management = input_data['test_data_management']
            test_automation = input_data['test_automation']
            metrics_and_kpis = input_data['metrics_and_kpis']
            documentation = input_data['documentation']

            prompt = f"""
            DORA Compliance Score for '{selected_testing_type}' testing type: {score}.
            Based on the following factors, provide specific recommendations to improve compliance in this area:
            - Frequency of Assessment: {frequency_of_assessment}
            - Critical Business Process Covered: {critical_business_process_covered}
            - Application Code Stack: {application_code_stack}
            - Production Data Usage: {production_data_usage}
            - Test Data Management: {test_data_management}
            - Test Automation: {test_automation}
            - Metrics and KPIs: {metrics_and_kpis}
            - Documentation Quality: {documentation}

            Additionally, identify any other DORA compliance testing types not currently covered. Explain the importance of each missing testing type for compliance and suggest specific actions the financial institution should take to address these gaps.
            """
            # Generate the recommendation
            try:
                response = openai.ChatCompletion.create(
                    engine="gpt-4",  # Ensure this matches your Azure deployment name
                    messages=[{"role": "user", "content": prompt}]
                )
                recommendation = response.choices[0].message['content'].strip()
            except openai.error.InvalidRequestError as e:
                recommendation = f"Error: {e.user_message}"
            except Exception as e:
                recommendation = f"An error occurred: {str(e)}"
            
            recommendations.append({
                "testing_type": selected_testing_type,
                "score": round(score,2),
                "recommendation": recommendation
            })
            
            # Introduce a delay to avoid API rate limits
            time.sleep(delay)
    
    return recommendations

# Calculate scores after entering inputs for all testing types
if st.button("Calculate DORA Compliance Score"):
    # Filter out any empty inputs before creating the DataFrame
    valid_inputs = [input_data for input_data in st.session_state.user_inputs if 'testing_type' in input_data]

    if not valid_inputs:
        st.error("Error: Some inputs are missing. Please fill in all fields.")
    else:
        # Prepare the DataFrame for predictions
        input_df = pd.DataFrame(valid_inputs)[feature_columns]

        # Encode categorical variables
        for col in feature_columns:
            input_df[col] = label_encoders[col].transform(input_df[col])  # Encode using the same encoder

        # Calculate DORA compliance scores for all testing types
        individual_scores = []
        for index, row in input_df.iterrows():
            predicted_score = best_rf_model.predict([row])[0]  # Predict the score
            individual_scores.append(predicted_score)

        # Calculate overall DORA compliance score
        overall_score = round(sum(individual_scores), 2)

        # Display results
        st.write("Overall DORA Compliance Score:", overall_score)

        # Get suggestions from the LLM (Azure OpenAI)
        if overall_score < 80:  # Example threshold for suggestions
            st.write("Generating recommendations for improvement...")
            recommendations = generate_compliance_recommendations(individual_scores, valid_inputs)
            
            for rec in recommendations:
                st.write(f"Testing Type: {rec['testing_type']}")
                st.write("Recommendation:", rec['recommendation'])
                st.write("---")
