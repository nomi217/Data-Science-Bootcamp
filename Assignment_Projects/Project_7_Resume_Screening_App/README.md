# Resume Screening App - Simple Version

A beginner-friendly resume classification app that uses machine learning to automatically categorize resumes and calculate fit scores.

## 🚀 Quick Start

1. **Install Python** (3.8 or higher)

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app:**
   
   **Option 1 - Command Line:**
   ```bash
   streamlit run app.py
   ```
   
   **Option 2 - Windows (Double-click):**
   - Double-click `run_app.bat`
   
   **Option 3 - Mac/Linux:**
   ```bash
   chmod +x run_app.sh
   ./run_app.sh
   ```

4. **Open your browser** and go to `http://localhost:8501`

## ⚠️ Important Note

**Always use `streamlit run app.py`** - NOT `python app.py`

The app will show an error message if you try to run it with `python app.py`

## 📁 What's in this project?

- `app.py` - The main application (everything in one file!)
- `requirements.txt` - Python packages needed
- `README.md` - This file

## 🎯 Features

- **Upload resumes** in PDF or DOCX format
- **Automatic classification** into job categories using real dataset
- **Real Dataset Training** - Uses Kaggle Resume Dataset (1000+ resumes)
- **Fallback Mode** - Works with sample data if dataset download fails
- **Job Categories** (from real dataset):
  - Data Science
  - Software Engineering  
  - Web Development
  - HR
  - Business Development
  - Health
  - Arts
  - Advocates
  - And more...
- **Fit score calculation** based on skills
- **Skill analysis** and visualization
- **Word cloud** generation

## 🔧 How it works

1. **Dataset Download**: Automatically downloads real resume dataset from Kaggle
2. **Text Extraction**: Extracts text from uploaded files
3. **Text Processing**: Cleans and preprocesses the text
4. **Machine Learning**: Uses a trained model to predict job category
5. **Skill Matching**: Matches skills against job requirements
6. **Visualization**: Shows results with charts and word clouds

## 📊 Dataset

The app prioritizes your **local UpdatedResumeDataSet.csv** file, then falls back to the Kaggle Resume Dataset, and finally uses sample data.

### Using Your Local Dataset
1. Place `UpdatedResumeDataSet.csv` in the same folder as `app.py`
2. The app will automatically detect and use it
3. No internet connection required!

### Test Your Local Dataset
```bash
python test_local_dataset.py
```

This will test your local CSV file and show you what categories are available.

### Test Kaggle Dataset Download
```bash
python test_dataset.py
```

This will test the Kaggle dataset download as a fallback.

## 🌐 Deploy Online

### Streamlit Cloud (Easiest)
1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Deploy!

### Other Options
- Hugging Face Spaces
- Heroku
- AWS

## 🛠️ Customization

Want to add more job categories or skills? Just edit the `categories` dictionary in `app.py`:

```python
self.categories = {
    'Your Category': ['skill1', 'skill2', 'skill3'],
    # ... existing categories
}
```

## ❓ Troubleshooting

**App won't start?**
- Make sure Python 3.8+ is installed
- Run `pip install -r requirements.txt`
- Try `streamlit run app.py`

**File upload not working?**
- Make sure file is PDF or DOCX
- Check file size (should be under 10MB)

**Low accuracy?**
- The app uses sample data for training
- For better results, train with real resume data

## 📚 Learning Resources

- [Streamlit Documentation](https://docs.streamlit.io)
- [Scikit-learn Tutorial](https://scikit-learn.org/stable/tutorial/)
- [NLTK Documentation](https://www.nltk.org/)

## 🤝 Contributing

Feel free to:
- Add new features
- Fix bugs
- Improve the UI
- Add more job categories

## 📄 License

This project is open source and free to use.

---

**Happy coding! 🎉**