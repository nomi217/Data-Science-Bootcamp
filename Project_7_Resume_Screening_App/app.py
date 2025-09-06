

import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import PyPDF2
import docx
from io import BytesIO
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px
import sys
import kagglehub
import os

# Download required NLTK data
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    st.warning("Note: NLTK data download may require internet connection")

# Page configuration
st.set_page_config(
    page_title="Resume Screening App",
    page_icon="📄",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .skill-tag {
        background-color: #e1f5fe;
        color: #01579b;
        padding: 0.25rem 0.5rem;
        border-radius: 1rem;
        font-size: 0.8rem;
        margin: 0.1rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

class SimpleResumeClassifier:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.model = LogisticRegression(random_state=42, max_iter=1000)
        self.label_encoder = None
        
        # Job categories and their skills (will be updated based on real dataset)
        self.categories = {
            'Data Science': ['python', 'machine learning', 'data analysis', 'pandas', 'numpy', 
                           'scikit-learn', 'tensorflow', 'pytorch', 'jupyter', 'statistics', 'sql'],
            'Software Engineering': ['java', 'python', 'javascript', 'react', 'spring', 'git', 
                                   'docker', 'aws', 'microservices', 'api', 'c++', 'c#'],
            'Web Development': ['html', 'css', 'javascript', 'react', 'bootstrap', 'php', 
                              'wordpress', 'responsive', 'frontend', 'backend', 'node.js'],
            'HR': ['recruitment', 'talent acquisition', 'employee relations', 'hr policies', 
                  'training', 'compensation', 'onboarding', 'workday'],
            'Business Development': ['sales', 'marketing', 'business development', 'partnerships', 
                                   'client relations', 'revenue', 'strategy', 'crm'],
            'Health': ['healthcare', 'medical', 'nursing', 'clinical', 'patient care', 
                      'health informatics', 'medical records', 'epic'],
            'Arts': ['design', 'graphic design', 'ui/ux', 'adobe', 'photoshop', 'illustrator',
                    'creativity', 'visual arts', 'digital art', 'branding'],
            'Advocates': ['law', 'legal', 'litigation', 'contracts', 'compliance', 'regulatory',
                         'court', 'attorney', 'counsel', 'legal research']
        }
    
    def preprocess_text(self, text):
        """Clean and preprocess text"""
        if not text:
            return ""
        
        # Convert to lowercase and remove special characters
        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize and remove stopwords
        tokens = word_tokenize(text)
        processed_tokens = []
        for token in tokens:
            if token not in self.stop_words and len(token) > 2:
                lemmatized = self.lemmatizer.lemmatize(token)
                processed_tokens.append(lemmatized)
        
        return ' '.join(processed_tokens)
    
    def extract_skills(self, text, category):
        """Extract skills for a specific category"""
        if category not in self.categories:
            return []
        
        text_lower = text.lower()
        matched_skills = []
        for skill in self.categories[category]:
            if skill in text_lower:
                matched_skills.append(skill)
        return matched_skills
    
    def calculate_fit_score(self, text, category, matched_skills):
        """Calculate fit score based on skills and text length"""
        if category not in self.categories:
            return 0.0
        
        # Skill matching score (70% weight)
        total_skills = len(self.categories[category])
        skill_score = (len(matched_skills) / total_skills) * 70
        
        # Text length score (30% weight)
        text_length = len(text.split())
        length_score = min((text_length / 500) * 30, 30)
        
        return min(skill_score + length_score, 100)
    
    def load_local_dataset(self):
        """Load the local UpdatedResumeDataSet.csv file"""
        try:
            #st.info("📥 Loading local dataset: UpdatedResumeDataSet.csv")
            
            # Check if the file exists
            if not os.path.exists("UpdatedResumeDataSet.csv"):
                st.error("❌ UpdatedResumeDataSet.csv not found in the current directory")
                st.info("💡 Please make sure the CSV file is in the same folder as app.py")
                return None, None
            
            # Load the CSV file
            df = pd.read_csv("UpdatedResumeDataSet.csv")
            #st.success(f"✅ Dataset loaded successfully! Shape: {df.shape}")
            
            # Show dataset info
            # st.info(f"📋 Columns: {df.columns.tolist()}")
            # st.info(f"📊 Categories: {df['Category'].value_counts().to_dict()}")
            
            # Check if we have the expected columns
            if 'Resume' in df.columns and 'Category' in df.columns:
                return df['Resume'].tolist(), df['Category'].tolist()
            else:
                st.error(f"Expected columns 'Resume' and 'Category' not found. Available columns: {df.columns.tolist()}")
                return None, None
                
        except Exception as e:
            st.error(f"Error loading local dataset: {str(e)}")
            st.info("💡 Tip: The app will use sample data as fallback")
            return None, None
    
    def download_dataset(self):
        """Download the real resume dataset from Kaggle (fallback method)"""
        try:
            #st.info("📥 Downloading resume dataset from Kaggle...")
            
            # Try different kagglehub methods
            path = None
            
            # Method 1: Try the download function
            if hasattr(kagglehub, 'download'):
                try:
                    path = kagglehub.download("gauravduttakiit/resume-dataset")
                except Exception as e:
                    st.warning(f"Method 1 failed: {str(e)}")
            
            # Method 2: Try the dataset_download function (older version)
            if path is None and hasattr(kagglehub, 'dataset_download'):
                try:
                    path = kagglehub.dataset_download("gauravduttakiit/resume-dataset")
                except Exception as e:
                    st.warning(f"Method 2 failed: {str(e)}")
            
            # Method 3: Try with kagglehub.load
            if path is None and hasattr(kagglehub, 'load'):
                try:
                    path = kagglehub.load("gauravduttakiit/resume-dataset")
                except Exception as e:
                    st.warning(f"Method 3 failed: {str(e)}")
            
            if path is None:
                st.error("Could not download dataset with any available method")
                return None, None
            
            # Look for CSV files in the downloaded path
            csv_files = []
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith('.csv'):
                        csv_files.append(os.path.join(root, file))
            
            if not csv_files:
                st.error("No CSV files found in the dataset")
                return None, None
            
            # Load the first CSV file
            df = pd.read_csv(csv_files[0])
            st.success(f"✅ Dataset loaded successfully! Shape: {df.shape}")
            
            # Check if we have the expected columns
            if 'Resume' in df.columns and 'Category' in df.columns:
                return df['Resume'].tolist(), df['Category'].tolist()
            else:
                st.error(f"Expected columns 'Resume' and 'Category' not found. Available columns: {df.columns.tolist()}")
                return None, None
                
        except Exception as e:
            st.error(f"Error downloading dataset: {str(e)}")
            st.info("💡 Tip: The app will use sample data as fallback")
            return None, None
    
    def create_sample_data(self):
        """Create sample training data as fallback"""
        sample_resumes = [
            "I am a Data Scientist with 5 years of experience in Python, Machine Learning, and Deep Learning. I have worked with Pandas, NumPy, and Scikit-learn on various data analysis projects.",
            "I am a Software Engineer specializing in Java, Spring Framework, and Microservices. I have experience with React, JavaScript, and Git for version control.",
            "I am a Web Developer with expertise in HTML, CSS, JavaScript, and React. I create responsive websites using Bootstrap and have worked with PHP and WordPress.",
            "I am an HR professional with experience in recruitment, talent acquisition, and employee relations. I manage HR policies and conduct training programs.",
            "I am a Business Development Manager with a strong background in sales, marketing, and client relations. I focus on revenue growth and strategic partnerships.",
            "I am a Healthcare professional with experience in medical records, patient care, and clinical operations. I have worked with Epic systems and health informatics.",
            "I am a Python developer with skills in data analysis, machine learning, and statistical modeling. I use Jupyter notebooks and SQL for data processing.",
            "I am a Full Stack Developer working with React, Node.js, and databases. I have experience in both frontend and backend development with JavaScript.",
            "I am an HR Manager responsible for recruitment, onboarding, and employee development. I handle compensation and benefits administration.",
            "I am a Data Analyst with strong skills in Python, SQL, and data visualization. I create reports and dashboards for business insights."
        ]
        
        sample_categories = [
            'Data Science', 'Software Engineering', 'Web Development', 'HR',
            'Business Development', 'Health', 'Data Science', 'Software Engineering',
            'HR', 'Data Science'
        ]
        
        return sample_resumes, sample_categories
    
    def update_categories_from_dataset(self, categories):
        """Update skill categories based on real dataset categories"""
        # Keep existing categories and add new ones if needed
        for category in categories:
            if category not in self.categories:
                # Add basic skills for new categories
                self.categories[category] = [category.lower(), 'experience', 'skills', 'professional']
    
    def train_model(self):
        """Train the model on local dataset first, then fallback to other methods"""
        # Try to load local dataset first
        resumes, categories = self.load_local_dataset()
        
        # If local dataset fails, try downloading from Kaggle
        if resumes is None or categories is None:
            st.info("🔄 Local dataset not found, trying to download from Kaggle...")
            resumes, categories = self.download_dataset()
        
        # If both fail, use sample data
        if resumes is None or categories is None:
            st.warning("⚠️ Using sample data as fallback. For better results, place UpdatedResumeDataSet.csv in the same folder as app.py")
            resumes, categories = self.create_sample_data()
        else:
            # Update categories based on real dataset
            self.update_categories_from_dataset(categories)
        
        # Preprocess text
        processed_resumes = [self.preprocess_text(resume) for resume in resumes]
        
        # Create features
        X = self.tfidf_vectorizer.fit_transform(processed_resumes)
        
        # Encode labels
        from sklearn.preprocessing import LabelEncoder
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(categories)
        
        # Train model
        self.model.fit(X, y)
        
        # Show training info
        #st.info(f"📊 Trained on {len(resumes)} resumes")
        #st.info(f"📋 Categories: {', '.join(self.label_encoder.classes_)}")
        
        return len(resumes)
    
    def predict(self, text):
        """Predict category for given text"""
        processed_text = self.preprocess_text(text)
        X = self.tfidf_vectorizer.transform([processed_text])
        
        prediction = self.model.predict(X)[0]
        probability = self.model.predict_proba(X)[0]
        
        category = self.label_encoder.inverse_transform([prediction])[0]
        confidence = max(probability) * 100
        
        # Extract skills and calculate fit score
        matched_skills = self.extract_skills(processed_text, category)
        fit_score = self.calculate_fit_score(processed_text, category, matched_skills)
        
        return category, confidence, matched_skills, fit_score

def extract_text_from_pdf(file_bytes):
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(BytesIO(file_bytes))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

def extract_text_from_docx(file_bytes):
    """Extract text from DOCX file"""
    try:
        doc = docx.Document(BytesIO(file_bytes))
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Error reading DOCX: {str(e)}")
        return ""

def create_wordcloud(text, title="Word Cloud"):
    """Create word cloud visualization"""
    if not text:
        return None
    
    wordcloud = WordCloud(
        width=800, height=400,
        background_color='white',
        max_words=100,
        colormap='viridis'
    ).generate(text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    return fig

def main():
    """Main application function"""
    # Header
    st.markdown('<h1 class="main-header">📄 Resume Screening App</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Initialize classifier
    if 'classifier' not in st.session_state:
        try:
            st.session_state.classifier = SimpleResumeClassifier()
            st.session_state.classifier.train_model()
            st.success("✅ Classifier ready!")
        except Exception as e:
            st.error(f"Error initializing classifier: {str(e)}")
            st.stop()
    
    classifier = st.session_state.classifier
    
    # Sidebar
    st.sidebar.title("🔧 Upload Resume")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a resume file",
        type=['pdf', 'docx'],
        help="Upload a PDF or DOCX resume file",
        label_visibility="visible"
    )
    
    # Main content
    if uploaded_file is not None:
        # Extract text based on file type
        file_extension = uploaded_file.name.lower().split('.')[-1]
        
        with st.spinner("Extracting text from resume..."):
            if file_extension == 'pdf':
                extracted_text = extract_text_from_pdf(uploaded_file.getvalue())
            elif file_extension == 'docx':
                extracted_text = extract_text_from_docx(uploaded_file.getvalue())
            else:
                st.error("Unsupported file format")
                return
        
        if not extracted_text:
            st.error("Failed to extract text from the file")
            return
        
        # Display extracted text
        st.subheader("📝 Extracted Resume Text")
        with st.expander("View extracted text", expanded=False):
            st.text_area("Resume Text", extracted_text, height=200, label_visibility="collapsed")
        
        # Make prediction
        with st.spinner("Analyzing resume..."):
            category, confidence, skills, fit_score = classifier.predict(extracted_text)
        
        # Display results
        st.markdown("---")
        
        # Results header
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Confidence", f"{confidence:.1f}%")
        
        with col2:
            st.metric("Fit Score", f"{fit_score:.1f}%")
        
        with col3:
            st.metric("Words", len(extracted_text.split()))
        
        # Display full predicted role prominently
        st.subheader(f"🎯 Predicted Role: {category}")
        
        # Detailed analysis
        tab1, tab2, tab3 = st.tabs(["📊 Analysis", "🔍 Skills", "☁️ Word Cloud"])
        
        with tab1:
            st.subheader("📊 Resume Analysis")
            
            # Fit score visualization
            st.subheader("Fit Score Breakdown")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Overall Match", f"{fit_score:.1f}%")
                
                # Progress bar
                progress_color = "green" if fit_score >= 80 else "orange" if fit_score >= 60 else "red"
                st.markdown(f"""
                <div style="background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem;">
                    <div style="background-color: {progress_color}; height: 20px; border-radius: 10px; 
                                width: {fit_score}%; transition: width 0.3s ease;"></div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.metric("Confidence", f"{confidence:.1f}%")
                st.metric("Matched Skills", len(skills))
            
            # Text statistics
            # st.subheader("📈 Text Statistics")
            # col1, col2, col3, col4 = st.columns(4)
            
            # with col1:
            #     st.metric("Characters", len(extracted_text))
            # with col2:
            #     st.metric("Words", len(extracted_text.split()))
            # with col3:
            #     st.metric("Sentences", len(extracted_text.split('.')))
            # with col4:
            #     st.metric("Paragraphs", len(extracted_text.split('\n\n')))
        
        with tab2:
            st.subheader("🔍 Skill Analysis")
            
            if skills:
                st.subheader("✅ Matched Skills")
                for skill in skills:
                    st.markdown(f'<span class="skill-tag">{skill}</span>', unsafe_allow_html=True)
                
                # Skill frequency chart
                if len(skills) > 1:
                    skill_counts = pd.Series(skills).value_counts()
                    fig = px.bar(
                        x=skill_counts.values,
                        y=skill_counts.index,
                        orientation='h',
                        title="Skill Frequency",
                        labels={'x': 'Frequency', 'y': 'Skills'}
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No specific skills detected for this category")
            
            # Show all possible skills for the category
            if category in classifier.categories:
                st.subheader(f"📋 Skills for {category}")
                all_skills = classifier.categories[category]
                for skill in all_skills:
                    if skill in skills:
                        st.markdown(f'<span class="skill-tag" style="background-color: #c8e6c9;">✓ {skill}</span>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<span class="skill-tag" style="background-color: #ffcdd2;">✗ {skill}</span>', unsafe_allow_html=True)
        
        with tab3:
            st.subheader("☁️ Word Cloud")
            
            # Create word cloud
            wordcloud_fig = create_wordcloud(extracted_text, "Resume Word Cloud")
            if wordcloud_fig:
                st.pyplot(wordcloud_fig)
            else:
                st.info("Not enough text to generate word cloud")
        
        # Recommendations
        st.subheader("💡 Recommendations")
        
        if confidence < 70:
            st.warning("⚠️ Low confidence prediction. The resume might not clearly fit any category.")
        
        if fit_score < 60:
            st.info("💡 Consider adding more relevant skills for the predicted role.")
        
        if len(extracted_text.split()) < 100:
            st.info("📝 Resume seems short. Consider adding more detailed information.")
        
        if len(skills) < 3:
            st.info("🎯 Add more specific skills relevant to the predicted role.")
    
    else:
        # Welcome screen
        st.markdown("""
        ## 🚀 Welcome to Resume Screening App
        
        This simple app uses machine learning to:
        
        - **📄 Extract text** from PDF and DOCX resume files
        - **🤖 Classify resumes** into job categories automatically  
        - **📊 Calculate fit scores** based on skill matching
        - **🔍 Analyze skills** and provide insights
        - **☁️ Generate visualizations** for better understanding
        
        ### How to use:
        1. Upload a resume file (PDF or DOCX) using the sidebar
        2. Wait for the analysis to complete
        3. View the predicted job category and fit score
        4. Explore detailed analysis in the tabs
        
        ### Supported Job Categories:
        """)
        
        # Display categories
        col1, col2, col3 = st.columns(3)
        categories = list(classifier.categories.keys())
        
        for i, category in enumerate(categories):
            with [col1, col2, col3][i % 3]:
                st.info(f"📌 {category}")
        
        # Dataset management
        st.subheader("📊 Dataset Management")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Retrain with Local Dataset"):
                with st.spinner("Loading and training on local dataset..."):
                    # Clear session state to force retraining
                    if 'classifier' in st.session_state:
                        del st.session_state.classifier
                    st.rerun()
        
        with col2:
            if st.button("🧪 Test with Sample Resume"):
                sample_text = """
                I am a Data Scientist with 5 years of experience in Python, Machine Learning, and Deep Learning. 
                I have worked with Pandas, NumPy, Scikit-learn, and TensorFlow on various data analysis projects. 
                I specialize in statistical modeling, data visualization, and building predictive models. 
                I have experience with SQL databases and Jupyter notebooks for data exploration and analysis.
                """
                
                # Make prediction
                category, confidence, skills, fit_score = classifier.predict(sample_text)
                
                st.success(f"Predicted: {category} (Confidence: {confidence:.1f}%, Fit Score: {fit_score:.1f}%)")
                st.write(f"Matched Skills: {', '.join(skills) if skills else 'None'}")
        
        # Show current model info
        # st.subheader("ℹ️ Current Model Info")
        # st.info(f"📋 Available Categories: {', '.join(classifier.label_encoder.classes_) if classifier.label_encoder else 'Not trained yet'}")
        # st.info(f"🔧 Model Type: Logistic Regression with TF-IDF")
        # st.info(f"📊 Features: {classifier.tfidf_vectorizer.max_features} most frequent words")

if __name__ == "__main__":
    # Check if running with streamlit
    if 'streamlit' in sys.modules:
        main()
    else:
        print("""
        ⚠️  Please run this app with Streamlit:
        
        streamlit run app.py
        
        Not: python app.py
        """)
        sys.exit(1)