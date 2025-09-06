import pandas as pd
import os

def test_local_dataset():
    """Test loading the local resume dataset"""
    print("🚀 Testing Local Resume Dataset")
    print("=" * 40)
    
    # Check if file exists
    if not os.path.exists("UpdatedResumeDataSet.csv"):
        print("❌ UpdatedResumeDataSet.csv not found in the current directory")
        print("💡 Please make sure the CSV file is in the same folder as this script")
        return False
    
    try:
        # Load the CSV file
        #print("📥 Loading UpdatedResumeDataSet.csv...")
        df = pd.read_csv("UpdatedResumeDataSet.csv")
        
        #print(f"✅ Dataset loaded successfully!")
        #print(f"📊 Shape: {df.shape}")
        #print(f"📋 Columns: {df.columns.tolist()}")
        
        # Show sample data
        #print("\n📝 Sample data:")
        #print(df.head())
        
        # Show categories if available
        # if 'Category' in df.columns:
        #     print(f"\n📂 Categories:")
        #     category_counts = df['Category'].value_counts()
        #     for category, count in category_counts.items():
        #         print(f"  - {category}: {count} resumes")
        
        # Show resume text sample
        # if 'Resume' in df.columns:
        #     print(f"\n📄 Sample resume text (first 200 characters):")
        #     print(f"'{df['Resume'].iloc[0][:200]}...'")
        
        #print("\n✅ Local dataset test completed successfully!")
        #print("You can now run the main app: streamlit run app.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure the file is named exactly 'UpdatedResumeDataSet.csv'")
        print("2. Check that the file is in the same directory as this script")
        print("3. Verify the file is not corrupted")
        return False

if __name__ == "__main__":
    test_local_dataset()
