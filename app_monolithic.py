import streamlit as st
import pandas as pd
import joblib

# Konfigurasi Halaman
st.set_page_config(page_title="Monolithic App: Student Predictor", layout="wide", page_icon="🎓")

# Inference Logic: Load Models
@st.cache_resource
def load_models():
    # Pastikan file .pkl berada di folder yang sama dengan file ini
    clf = joblib.load('clf_model.pkl')
    reg = joblib.load('reg_model.pkl')
    return clf, reg

try:
    clf_model, reg_model = load_models()
except FileNotFoundError:
    st.error("File model (clf_model.pkl atau reg_model.pkl) tidak ditemukan. Pastikan sudah menjalankan pipeline training.")
    st.stop()

st.title("🎓 Student Performance & Placement Predictor")
st.markdown("Aplikasi Monolithic: Logika prediksi dan antarmuka pengguna berada di dalam satu sistem (server) yang sama.")

# UI/UX: Sidebar & Form
st.sidebar.header("📝 Input Data Mahasiswa")

with st.sidebar.form("monolithic_form"):
    st.subheader("Informasi Dasar")
    gender = st.selectbox("Gender", ['Male', 'Female'])
    branch = st.selectbox("Branch", ['CSE', 'IT', 'ECE', 'ME', 'EEE']) 
    city_tier = st.selectbox("City Tier", ['Tier 1', 'Tier 2', 'Tier 3'])
    family_income_level = st.selectbox("Family Income Level", ['Low', 'Medium', 'High'])
    
    st.subheader("Performa Akademik")
    cgpa = st.slider("CGPA", 0.0, 10.0, 7.5)
    tenth_percentage = st.slider("10th Grade (%)", 0.0, 100.0, 80.0)
    twelfth_percentage = st.slider("12th Grade (%)", 0.0, 100.0, 80.0)
    backlogs = st.number_input("Backlogs", min_value=0, max_value=10, value=0)
    attendance_percentage = st.slider("Attendance (%)", 0.0, 100.0, 85.0)
    
    st.subheader("Skill & Pengalaman")
    study_hours_per_day = st.number_input("Study Hours per Day", min_value=0.0, max_value=24.0, value=4.0)
    projects_completed = st.number_input("Projects Completed", min_value=0, max_value=20, value=2)
    internships_completed = st.number_input("Internships Completed", min_value=0, max_value=10, value=1)
    hackathons_participated = st.number_input("Hackathons Participated", min_value=0, max_value=20, value=0)
    certifications_count = st.number_input("Certifications Count", min_value=0, max_value=20, value=1)
    
    st.markdown("**Skill Ratings (0-100)**")
    coding_skill_rating = st.slider("Coding Skill", 0, 100, 75)
    communication_skill_rating = st.slider("Communication Skill", 0, 100, 75)
    aptitude_skill_rating = st.slider("Aptitude Skill", 0, 100, 75)
    
    st.subheader("Gaya Hidup & Lainnya")
    sleep_hours = st.number_input("Sleep Hours", min_value=0.0, max_value=24.0, value=7.0)
    stress_level = st.slider("Stress Level", 1, 10, 5) 
    part_time_job = st.selectbox("Part-time Job", ['Yes', 'No'])
    internet_access = st.selectbox("Internet Access", ['Yes', 'No'])
    extracurricular_involvement = st.selectbox("Extracurricular", ['Yes', 'No', 'None'])

    # Tombol eksekusi form
    submitted = st.form_submit_button("Predict")

# Inference Logic & Data Visualization
if submitted:
    # Siapkan DataFrame dari input
    input_data = pd.DataFrame({
        "gender": [gender], "branch": [branch], "cgpa": [cgpa], 
        "tenth_percentage": [tenth_percentage], "twelfth_percentage": [twelfth_percentage], 
        "backlogs": [backlogs], "study_hours_per_day": [study_hours_per_day], 
        "attendance_percentage": [attendance_percentage], "projects_completed": [projects_completed], 
        "internships_completed": [internships_completed], "coding_skill_rating": [coding_skill_rating], 
        "communication_skill_rating": [communication_skill_rating], "aptitude_skill_rating": [aptitude_skill_rating], 
        "hackathons_participated": [hackathons_participated], "certifications_count": [certifications_count], 
        "sleep_hours": [sleep_hours], "stress_level": [stress_level], 
        "part_time_job": [part_time_job], "family_income_level": [family_income_level], 
        "city_tier": [city_tier], "internet_access": [internet_access], 
        "extracurricular_involvement": [extracurricular_involvement]
    })
    
    st.divider()
    
    # Eksekusi Prediksi
    placement_pred = clf_model.predict(input_data)[0]
    
    if placement_pred == 1:
        st.success("### 🎉 Placement Status: Placed")
        
        # Prediksi Regresi (Gaji)
        salary_pred = reg_model.predict(input_data)[0]
        
        # UI/UX: Data Visualization (Metric & Progress Bar)
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Estimated Salary (LPA)", value=f"{salary_pred:.2f}")
            
        with col2:
            # Menampilkan probabilitas jika model mendukung (Logistic Regression mendukung ini)
            try:
                prob = clf_model.predict_proba(input_data)[0][1] * 100
                st.write("**Confidence/Probability:**")
                st.progress(int(prob))
                st.caption(f"Tingkat keyakinan model: {prob:.1f}%")
            except:
                st.info("Visualisasi probabilitas tidak tersedia untuk model ini.")
                
    else:
        st.error("### ❌ Placement Status: Not Placed")
        st.warning("Saran: Fokus pada peningkatan kompetensi (CGPA, Coding Skill) dan hindari mata kuliah yang mengulang (backlogs).")