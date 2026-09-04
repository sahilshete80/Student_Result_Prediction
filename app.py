import streamlit as st
import pickle
import numpy as np
import time

# Page configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling, buttons, and animations
st.markdown("""
    <style>
    /* Main background and font adjustments */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Card Container Styling */
    .css-card {
        background-color: #ffffff;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
    }

    /* Custom Input Headers */
    .input-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 12px;
    }

    /* Custom Predict Button Styling */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        font-size: 1.1rem;
        font-weight: 600;
        padding: 0.6rem 2rem;
        border-radius: 10px;
        border: none;
        box-shadow: 0 4px 14px 0 rgba(124, 58, 237, 0.39);
        transition: all 0.3s ease;
        width: 100%;
    }
    
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px 0 rgba(124, 58, 237, 0.5);
        color: white;
    }

    div.stButton > button:first-child:active {
        transform: translateY(1px);
    }

    /* Prediction Box Styling */
    .result-box {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 10px 25px -5px rgba(16, 185, 129, 0.4);
        animation: fadeIn 0.5s ease-in-out;
    }

    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(10px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the K-Neighbors Classifier model."""
    try:
        with open('model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("⚠️ `model.pkl` file not found. Please ensure it is located in the same directory.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def trigger_confetti():
    """Generates an attractive celebratory visual effect."""
    st.balloons()


def main():
    model = load_model()

    # Sidebar Header
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3135/3135755.png", width=100)
        st.title("Grade Analytics")
        st.markdown("Predict overall performance based on individual subject marks.")
        st.markdown("---")
        st.info("💡 **Model:** K-Neighbors Classifier\n\nEnter scores ranging from **0 to 100** for each subject.")

    # App Header
    st.title("🎓 Student Marks & Result Predictor")
    st.markdown("Fill in the academic scores below to calculate the predicted score/total.")

    st.markdown("---")

    # Feature inputs based on the pickle model structure
    # Subjects: Hindi, English, Science, Maths, History, Geography, Total
    subjects = ["Hindi", "English", "Science", "Maths", "History", "Geography", "Total"]

    st.markdown("### 📝 Enter Subject Marks")
    
    # Input Layout organized into two clean columns
    col1, col2 = st.columns(2)

    input_data = {}
    
    with col1:
        input_data["Hindi"] = st.number_input("Hindi Score", min_value=0, max_value=100, value=75, step=1)
        input_data["English"] = st.number_input("English Score", min_value=0, max_value=100, value=80, step=1)
        input_data["Science"] = st.number_input("Science Score", min_value=0, max_value=100, value=85, step=1)
        input_data["Maths"] = st.number_input("Maths Score", min_value=0, max_value=100, value=90, step=1)

    with col2:
        input_data["History"] = st.number_input("History Score", min_value=0, max_value=100, value=70, step=1)
        input_data["Geography"] = st.number_input("Geography Score", min_value=0, max_value=100, value=78, step=1)
        
        # Calculate calculated total automatically or let user override
        calculated_sum = sum([input_data[sub] for sub in ["Hindi", "English", "Science", "Maths", "History", "Geography"]])
        input_data["Total"] = st.number_input("Total Aggregated Score", min_value=0, max_value=600, value=calculated_sum, step=1)

    st.markdown("<br>", unsafe_allow_html=True)

    # Centered Predict Button Section
    _, btn_col, _ = st.columns([1, 2, 1])

    with btn_col:
        predict_btn = st.button("🚀 Generate Prediction")

    if predict_btn:
        if model is not None:
            # Prepare feature array in the exact order required by the model
            features = np.array([[input_data[sub] for sub in subjects]])

            # Loading effect
            with st.spinner("Analyzing performance pattern..."):
                time.sleep(0.6)  # Brief delay to enhance UX feel
                prediction = model.predict(features)[0]

            # Success Trigger & Effects
            trigger_confetti()

            st.markdown("<br>", unsafe_allow_html=True)
            
            # Display Prediction Result Card
            st.markdown(
                f"""
                <div class="result-box">
                    <h3 style="margin:0; font-size: 1.2rem; opacity: 0.9;">Predicted Result</h3>
                    <h1 style="margin: 10px 0 0 0; font-size: 2.8rem; font-weight: 700;">{prediction}</h1>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Extra Metrics Display
            st.markdown("### 📊 Summary Overview")
            m_col1, m_col2, m_col3 = st.columns(3)
            
            avg_score = round(calculated_sum / 6, 2)
            m_col1.metric(label="Calculated Sum", value=f"{calculated_sum} / 600")
            m_col2.metric(label="Average Marks", value=f"{avg_score}%")
            m_col3.metric(label="Model Verdict", value=str(prediction))


if __name__ == "__main__":
    main()
