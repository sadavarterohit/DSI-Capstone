import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from openai import OpenAI
from dotenv import load_dotenv
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import whisper
import tempfile
import io
from pydub import AudioSegment

# Set page config
st.set_page_config(
    page_title="Fraud Detection System", 
    page_icon="🛡️",
    layout="wide"
)

# Global variables to store the trained models
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False

# EXACT NOTEBOOK WORKFLOW - Step 1: Load Dataset
@st.cache_data
def load_dataset():
    """Load the dataset - exact copy from notebook"""
    df = pd.read_csv('cleaned_conversations.csv')
    return df

# EXACT NOTEBOOK WORKFLOW - Step 2: Text Preprocessing Functions
@st.cache_resource
def init_nltk():
    """Download NLTK data - with SSL certificate handling"""
    import ssl
    
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    
    # Download NLTK data with error handling
    try:
        nltk.download('punkt', quiet=True)
    except:
        pass
    
    try:
        nltk.download('stopwords', quiet=True)
    except:
        pass
        
    try:
        nltk.download('punkt_tab', quiet=True)
    except:
        pass
    
    # Initialize stemmer and stopwords
    stemmer = PorterStemmer()
    
    # Try to load stopwords, fall back to basic list if download fails
    try:
        stop_words = set(stopwords.words('english'))
    except:
        # Basic English stopwords as fallback
        stop_words = {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
            'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
            'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
            'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
            'while', 'of', 'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after',
            'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
            'further', 'then', 'once'
        }
    
    return stemmer, stop_words

def clean_text(text):
    """Clean and preprocess text data - EXACT copy from notebook"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def tokenize_and_process(text, stemmer, stop_words):
    """Tokenize text, remove stopwords, and apply stemming - with fallback tokenization"""
    # Tokenize with fallback if NLTK punkt is not available
    try:
        tokens = word_tokenize(text)
    except:
        # Simple fallback tokenization
        import re
        tokens = re.findall(r'\b\w+\b', text.lower())
    
    # Remove stopwords and apply stemming
    processed_tokens = [
        stemmer.stem(token) for token in tokens 
        if token not in stop_words and len(token) > 2
    ]
    
    return processed_tokens

# EXACT NOTEBOOK WORKFLOW - Step 3: Preprocess all texts
@st.cache_data
def preprocess_dataframe(_df, _stemmer, _stop_words):
    """Preprocess the entire dataframe - EXACT copy from notebook"""
    df = _df.copy()
    
    # Apply text preprocessing to the 'text' column
    df['cleaned_text'] = df['perp_text_en'].apply(clean_text)
    df['tokenized_text'] = df['cleaned_text'].apply(lambda x: tokenize_and_process(x, _stemmer, _stop_words))
    
    # Create a processed text column (tokens joined back into strings)
    df['processed_text'] = df['tokenized_text'].apply(lambda x: ' '.join(x))
    
    return df

# EXACT NOTEBOOK WORKFLOW - Step 4: Load BERT Model and Generate Embeddings
@st.cache_resource
def load_and_generate_embeddings(df_processed):
    """EXACT copy from notebook - do it all in one step like the notebook"""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        embeddings = model.encode(df_processed['processed_text'].tolist(), convert_to_tensor=False)
        df_processed['bert_embeddings'] = [row for row in embeddings]
        # Create df2 exactly like notebook
        df2 = df_processed[['bert_embeddings', 'fraud_type']]
        return df2, embeddings, model
    except ImportError:
        # Fallback to simple TF-IDF if BERT fails
        st.warning("⚠️ BERT model unavailable, using TF-IDF fallback")
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(max_features=384, stop_words='english')
        embeddings = vectorizer.fit_transform(df_processed['processed_text']).toarray()
        df_processed['bert_embeddings'] = [row for row in embeddings]
        # Create df2 exactly like notebook
        df2 = df_processed[['bert_embeddings', 'fraud_type']]
        return df2, embeddings, vectorizer

# EXACT NOTEBOOK WORKFLOW - Step 5: Train XGBoost Model
@st.cache_data
def train_xgboost_model(_df2, _embeddings):
    """Train XGBoost model - EXACT copy from notebook"""
    # Convert BERT embeddings to numpy array
    X = np.array(_embeddings)
    
    # Handle NaN values in fraud_type - exact copy from notebook error handling
    df_clean = _df2.dropna(subset=['fraud_type'])
    valid_indices = _df2.dropna(subset=['fraud_type']).index
    X_clean = X[valid_indices]
    
    # Prepare target variable - encode fraud_type to numeric labels
    le = LabelEncoder()
    y = le.fit_transform(df_clean['fraud_type'].values)
    
    # Split the data - EXACT parameters from notebook
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    
    # Train XGBoost - EXACT parameters from notebook
    n_classes = len(le.classes_)
    if n_classes == 2:
        xgb_model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=42,
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8
        )
    else:
        xgb_model = xgb.XGBClassifier(
            objective='multi:softprob',
            eval_metric='mlogloss',
            random_state=42,
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8
        )
    
    xgb_model.fit(X_train, y_train)
    
    return xgb_model, le, X_train, X_test, y_train, y_test

# AUDIO PROCESSING FUNCTIONS - Using Whisper for transcription
@st.cache_resource
def load_whisper_model():
    """Load Whisper model - using base model for good accuracy/speed balance"""
    return whisper.load_model("base")

def transcribe_audio(audio_file, whisper_model):
    """Transcribe audio file using Whisper"""
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            # Convert audio to proper format if needed
            if audio_file.type not in ['audio/wav', 'audio/wave']:
                # Convert to WAV format using pydub
                audio_segment = AudioSegment.from_file(io.BytesIO(audio_file.getvalue()))
                audio_segment.export(tmp_file.name, format="wav")
            else:
                tmp_file.write(audio_file.getvalue())
            
            # Transcribe with Whisper
            result = whisper_model.transcribe(tmp_file.name, language="en")
            
            # Clean up temp file
            os.unlink(tmp_file.name)
            
            return result["text"]
            
    except Exception as e:
        return f"❌ Transcription Error: {str(e)}"

# EXACT NOTEBOOK WORKFLOW - OpenAI Explanation Function
def explain_fraud_type(text, fraud_type):
    """EXACT copy of explain_fraud_type function from notebook"""
    # Load API key from code/.env file - exact same as notebook
    load_dotenv('.env')
    
    # Debug: Check if API key is loaded
    api_key = os.getenv('API_KEY')
    if not api_key:
        return "❌ Error: API key not found in .env file"
    
    # Remove quotes if present (common issue with .env files)
    if api_key.startswith('"') and api_key.endswith('"'):
        api_key = api_key[1:-1]
    
    client = OpenAI(api_key=api_key)
    
    # Create prompt for OpenAI
    prompt = f"""
    Analyze this text and explain why it represents "{fraud_type}".
    
    Text: "{text}"
    Fraud Type: {fraud_type}
    
    Provide a clear, concise explanation covering:
    1. Key indicators that point to this fraud type
    2. Specific suspicious words or phrases
    3. Why this classification makes sense
    
    Keep it under 150 words.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a fraud detection expert. Explain fraud patterns clearly and concisely."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.3
        )
        
        explanation = response.choices[0].message.content.strip()
        return explanation
        
    except Exception as e:
        return f"❌ Error: {e}"

# MAIN STREAMLIT APP - Following EXACT notebook workflow
def main():
    st.title("🛡️ Fraud Detection System")
    st.markdown("### 📝 Text Analysis + 🎤 Audio Transcription")
    st.markdown("*Following EXACT BERT+XGB_TeleAnti.ipynb Workflow with Whisper Audio Support*")
    
    # Step 1: Initialize and load everything following notebook order
    if not st.session_state.models_loaded:
        with st.spinner("🔄 Executing Notebook Workflow Step-by-Step..."):
            
            # Step 1: Load Dataset (Cell 1)
            st.write("📊 Step 1: Loading dataset...")
            df = load_dataset()
            st.success(f"✅ Loaded {len(df)} records")
            
            # Step 2: Initialize NLTK (Cell 2 part 1)
            st.write("🔧 Step 2: Initializing NLTK...")
            stemmer, stop_words = init_nltk()
            st.success("✅ NLTK initialized")
            
            # Step 3: Preprocess Data (Cell 2 part 2) 
            st.write("📝 Step 3: Preprocessing text data...")
            df_processed = preprocess_dataframe(df, stemmer, stop_words)
            st.success("✅ Text preprocessing completed")
            
            # Step 4 & 5: Load BERT Model and Generate Embeddings (Cell 4 - EXACT like notebook)
            st.write("🧠 Step 4: Loading BERT model and generating embeddings...")
            df2, embeddings, bert_model = load_and_generate_embeddings(df_processed)
            st.success("✅ BERT model loaded and embeddings generated")
            
            # Step 6: Train XGBoost (Cell 7-8)
            st.write("🚀 Step 6: Training XGBoost model...")
            xgb_model, le, X_train, X_test, y_train, y_test = train_xgboost_model(df2, embeddings)
            st.success("✅ XGBoost model trained")
            
            # Store in session state
            st.session_state.stemmer = stemmer
            st.session_state.stop_words = stop_words
            st.session_state.bert_model = bert_model
            st.session_state.xgb_model = xgb_model
            st.session_state.le = le
            st.session_state.models_loaded = True
            
        st.success("🎯 **All Models Ready! Following Exact Notebook Workflow**")
    
    # Main Interface
    st.markdown("---")
    
    # Input Method Selection
    st.subheader("🎯 Choose Input Method")
    input_method = st.radio(
        "Select how you want to provide the text for analysis:",
        ["📝 Type Text", "🎤 Upload Audio"],
        horizontal=True
    )
    
    # Create columns for input and info
    col1, col2 = st.columns([2, 1])
    
    with col1:
        user_text = ""
        
        if input_method == "📝 Type Text":
            st.subheader("📝 Enter Text to Analyze")
            user_text = st.text_area(
                "Enter suspicious text:",
                placeholder="Type or paste the text you want to analyze for fraud...",
                height=150,
                key="user_input"
            )
            
        else:  # Audio input
            st.subheader("🎤 Upload Audio for Analysis")
            
            # Load Whisper model if not already loaded
            if 'whisper_model' not in st.session_state:
                with st.spinner("🔄 Loading Whisper model for audio transcription..."):
                    st.session_state.whisper_model = load_whisper_model()
                st.success("✅ Whisper model loaded!")
            
            # Audio file upload
            uploaded_audio = st.file_uploader(
                "Choose an audio file:",
                type=['wav', 'mp3', 'mp4', 'm4a', 'ogg', 'flac'],
                help="Upload audio file containing suspicious conversation or message"
            )
            
            if uploaded_audio is not None:
                # Display audio player
                st.audio(uploaded_audio, format=f'audio/{uploaded_audio.type.split("/")[1]}')
                
                # Transcribe button
                if st.button("🔄 Transcribe Audio", type="secondary"):
                    with st.spinner("🎤 Transcribing audio... This may take a moment."):
                        transcribed_text = transcribe_audio(uploaded_audio, st.session_state.whisper_model)
                        st.session_state.transcribed_text = transcribed_text
                
                # Show transcribed text if available
                if hasattr(st.session_state, 'transcribed_text'):
                    st.subheader("📝 Transcribed Text:")
                    user_text = st.text_area(
                        "Review and edit the transcribed text if needed:",
                        value=st.session_state.transcribed_text,
                        height=150,
                        key="transcribed_input",
                        help="You can edit this text before analyzing"
                    )
                    
                    if st.session_state.transcribed_text.startswith("❌"):
                        st.error("Audio transcription failed. Please try a different audio file.")
                        user_text = ""
        
    # Analyze button - outside columns so it spans full width
    st.markdown("---")
    
    analyze_button_text = "🔍 Analyze Text" if input_method == "📝 Type Text" else "🔍 Analyze Transcribed Text"
    
    if st.button(analyze_button_text, type="primary", disabled=not user_text.strip()):
        if user_text.strip():
            with st.spinner("🔄 Processing through exact notebook pipeline..."):
                
                # Add input method info to results
                input_source = "Audio (Transcribed)" if input_method == "🎤 Upload Audio" else "Text Input"
                
                # EXACT notebook workflow for new text:
                
                # 1. Clean text (exact function from notebook)
                cleaned_text = clean_text(user_text)
                
                # 2. Tokenize and process (exact function from notebook)
                tokens = tokenize_and_process(cleaned_text, 
                                            st.session_state.stemmer, 
                                            st.session_state.stop_words)
                
                # 3. Create processed text (exact from notebook)
                processed_text = ' '.join(tokens)
                
                # 4. Generate embedding (BERT or TF-IDF fallback)
                try:
                    embedding = st.session_state.bert_model.encode([processed_text], convert_to_tensor=False)
                except AttributeError:
                    # TF-IDF fallback
                    embedding = st.session_state.bert_model.transform([processed_text]).toarray()
                
                # 5. Predict using XGBoost (exact from notebook)
                prediction = st.session_state.xgb_model.predict(embedding)[0]
                prediction_proba = st.session_state.xgb_model.predict_proba(embedding)[0]
                
                # 6. Get fraud type name (exact from notebook)
                fraud_type = st.session_state.le.inverse_transform([prediction])[0]
                confidence = prediction_proba[prediction]
                
                # Store results
                st.session_state.result = {
                    'original_text': user_text,
                    'cleaned_text': cleaned_text,
                    'processed_text': processed_text,
                    'fraud_type': fraud_type,
                    'confidence': confidence,
                    'all_probabilities': prediction_proba,
                    'all_classes': st.session_state.le.classes_,
                    'input_source': input_source
                }
    
    with col2:
        st.subheader("ℹ️ Model Info")
        if st.session_state.models_loaded:
            whisper_status = "✅ Loaded" if 'whisper_model' in st.session_state else "⏳ Not loaded"
            st.info(f"""
            **✅ Loaded Models:**
            - BERT: bge-small-en-v1.5
            - XGBoost: Multi-class
            - Classes: {len(st.session_state.le.classes_)}
            - OpenAI: GPT-3.5-turbo
            - Whisper: {whisper_status}
            """)
        else:
            st.warning("Models not loaded yet")
        
        # Audio format info
        if input_method == "🎤 Upload Audio":
            st.info("""
            **📁 Supported Formats:**
            - WAV, MP3, MP4
            - M4A, OGG, FLAC
            - Max size: 200MB
            """)
            
            st.warning("""
            **⚠️ Audio Tips:**
            - Clear speech works best
            - Minimize background noise
            - English language only
            - Keep under 30 seconds for faster processing
            """)
    
    # Display Results
    if hasattr(st.session_state, 'result'):
        result = st.session_state.result
        
        st.markdown("---")
        st.subheader("🎯 Fraud Detection Results")
        
        # Input source indicator
        st.info(f"📊 **Input Source:** {result.get('input_source', 'Text Input')}")
        
        # Main result display
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            # Color code based on fraud type
            if 'no fraud' in str(result['fraud_type']).lower():
                st.success(f"✅ **Result:** {result['fraud_type']}")
            else:
                st.error(f"⚠️ **Result:** {result['fraud_type']}")
        
        with col2:
            st.metric("Confidence", f"{result['confidence']:.1%}")
        
        with col3:
            if st.button("🤖 Explain", type="secondary"):
                st.session_state.show_explanation = True
        
        # Show processing pipeline
        with st.expander("🔍 View Processing Pipeline"):
            st.write("**Original Text:**")
            st.text(result['original_text'][:200] + "..." if len(result['original_text']) > 200 else result['original_text'])
            
            st.write("**Cleaned Text:**")
            st.text(result['cleaned_text'][:200] + "..." if len(result['cleaned_text']) > 200 else result['cleaned_text'])
            
            st.write("**Processed Text:**")
            st.text(result['processed_text'][:200] + "..." if len(result['processed_text']) > 200 else result['processed_text'])
        
        # All probabilities
        st.subheader("📊 All Class Probabilities")
        prob_df = pd.DataFrame({
            'Fraud Type': result['all_classes'],
            'Probability': result['all_probabilities']
        }).sort_values('Probability', ascending=False)
        
        st.bar_chart(prob_df.set_index('Fraud Type')['Probability'])
        
        # AI Explanation (exact from notebook)
        if hasattr(st.session_state, 'show_explanation') and st.session_state.show_explanation:
            st.subheader("🧠 AI Explanation")
            
            with st.spinner("🤖 Generating explanation..."):
                explanation = explain_fraud_type(result['original_text'], result['fraud_type'])
            
            st.markdown(f"""
            <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; border-left: 4px solid #1f77b4;">
                <h4>🤖 AI Analysis:</h4>
                <p>{explanation}</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("❌ Hide Explanation"):
                st.session_state.show_explanation = False
                st.rerun()
    
    # Example texts and audio scenarios
    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("💡 Example Texts (Try These)")
    
    with col2:
        st.subheader("🎤 Audio Scenarios to Test")
        st.markdown("""
        **Record or find audio with these scenarios:**
        - 📞 Fake tech support calls
        - 💰 Lottery/prize scam messages  
        - 💕 Romance scam conversations
        - 🏦 Phishing bank calls
        - ✅ Legitimate business calls
        """)
    
    examples = [
        ("Lottery Scam", "Congratulations! You've won a lottery of $1,000,000. To claim your prize, please provide your bank details."),
        ("Romance Scam", "My dear, I am stuck in another country and need money for my flight home. Please send $500."),
        ("Tech Support", "Your computer has been infected! Call this number immediately: 1-800-SCAM"),
        ("Legitimate", "Hi, this is a reminder about your dentist appointment tomorrow at 2 PM.")
    ]
    
    cols = st.columns(4)
    for i, (title, text) in enumerate(examples):
        with cols[i]:
            if st.button(f"📋 {title}", key=f"ex_{i}"):
                st.session_state.example_selected = text
                st.rerun()
    
    # Show selected example
    if hasattr(st.session_state, 'example_selected'):
        st.text_area("Selected Example (copy to input above):", 
                    value=st.session_state.example_selected, 
                    height=80, key="example_display")

if __name__ == "__main__":
    main()
