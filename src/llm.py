# src/llm.py - LLM interface using Ollama
import os
import requests
import subprocess
import time
import platform
import pandas as pd
from src.prompts import get_query_understanding_prompt

# Ollama API URL
OLLAMA_API_URL = "http://localhost:11434/api/generate"

# Model to use - can be changed to other models if needed
MODEL_NAME = "qwen:4b"  # Fallbacks: "llama2:7b", "mistral:7b", "phi:latest"

def check_ollama_installed():
    """Check if Ollama is installed on the system."""
    try:
        # Check if the ollama command exists
        subprocess.run(
            ["which", "ollama"] if platform.system() != "Windows" else ["where", "ollama"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        return True
    except subprocess.CalledProcessError:
        return False
    except Exception:
        return False

def check_model_pulled():
    """Check if the model has been pulled in Ollama."""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            return any(model.get("name") == MODEL_NAME for model in models)
        return False
    except requests.RequestException:
        return False
    except Exception:
        return False

def start_ollama_server():
    """Start the Ollama server if it's not already running."""
    try:
        # Check if Ollama server is running
        requests.get("http://localhost:11434/api/version")
        print("Ollama server is already running.")
        return True
    except requests.RequestException:
        # Start Ollama server
        print("Starting Ollama server...")
        try:
            # Start in background
            if platform.system() == "Windows":
                subprocess.Popen(["ollama", "serve"], 
                                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
            else:
                subprocess.Popen(["ollama", "serve"], 
                                stdout=subprocess.PIPE, 
                                stderr=subprocess.PIPE)
            
            # Wait for server to start
            max_retries = 5
            for i in range(max_retries):
                time.sleep(2)  # Wait a bit
                try:
                    requests.get("http://localhost:11434/api/version")
                    print("Ollama server started successfully.")
                    return True
                except requests.RequestException:
                    if i == max_retries - 1:
                        print("Failed to start Ollama server.")
                        return False
                    print(f"Waiting for Ollama server to start (attempt {i+1}/{max_retries})...")
            
            return False
        except Exception as e:
            print(f"Error starting Ollama server: {e}")
            return False

def ensure_ollama_running():
    """Ensure Ollama is running with the required model."""
    # Skip if we're in simulation mode
    if os.environ.get("SIMULATE_LLM", "").lower() == "true":
        return True
        
    # Check if Ollama is installed
    if not check_ollama_installed():
        print("⚠️ Ollama is not installed. Using fallback mode.")
        os.environ["SIMULATE_LLM"] = "true"
        return False
    
    # Start Ollama server if needed
    if not start_ollama_server():
        print("⚠️ Failed to start Ollama server. Using fallback mode.")
        os.environ["SIMULATE_LLM"] = "true"
        return False
    
    return True

def process_query(query, df=None):
    """Process a natural language query through Ollama API.
    
    Args:
        query: The natural language query
        df: Optional dataframe for context
        
    Returns:
        Executable Python code
    """
    # Check if we should use the simulated mode
    if os.environ.get("SIMULATE_LLM", "").lower() == "true":
        return get_simulated_response(query, df)
    
    # Ensure Ollama is running
    ensure_ollama_running()
    
    # Add dataframe context if provided
    df_info = ""
    if df is not None:
        # Get column information
        columns = df.columns.tolist()
        dtypes = df.dtypes.to_dict()
        dtypes_str = {col: str(dtype) for col, dtype in dtypes.items()}
        
        # Sample values for each column
        sample_values = {}
        for col in columns:
            try:
                # Get non-null sample values if possible
                sample = df[col].dropna().head(3).tolist()
                if len(sample) > 0:
                    sample_values[col] = sample
            except:
                continue
        
        # Create dataframe context
        df_info = f"""
Available columns: {columns}
Column types: {dtypes_str}
Sample values:
{sample_values}
        """
    
    # Get the prompt
    prompt = get_query_understanding_prompt(query, df_info)
    
    # Prepare the request payload
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 512,
            "stop": ["```"]
        }
    }
    
    try:
        # Send the request to Ollama API
        response = requests.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Parse the response
        result = response.json()
        generated_text = result.get("response", "").strip()
        
        # Clean up the response to get just the executable code
        code_lines = []
        for line in generated_text.split('\n'):
            line = line.strip()
            if line and not line.startswith('Output:') and not line.startswith('```'):
                code_lines.append(line)
        
        code = '\n'.join(code_lines)
        
        # Add import statements if they're not already there
        if 'import pandas as pd' not in code:
            code = 'import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nfrom scipy import stats\n\n' + code
        
        # If it doesn't use 'df', assume it's a standalone fragment and wrap it
        if 'df' not in code:
            code = 'result = ' + code
        else:
            # Add a result line if there isn't one
            if not any(line.strip().startswith('result =') for line in code.split('\n')):
                code += '\n\n# Store result\nresult = df'
        
        return code
    
    except requests.RequestException as e:
        # Handle API request errors
        error_msg = f"Error communicating with Ollama API: {str(e)}"
        print(error_msg)
        return get_simulated_response(query, df)

def get_simulated_response(query, df=None):
    """Generate a simulated response when Ollama is not available.
    
    Args:
        query: The natural language query
        df: Optional dataframe for context
        
    Returns:
        Simulated Python code based on common patterns
    """
    # Set of pre-built responses for common queries
    query = query.lower()
    
    # Basic patterns to match
    if "show" in query and "distribution" in query:
        # Show distribution
        column = None
        if df is not None:
            # Try to find a numeric column
            for col in df.columns:
                if df[col].dtype in ['int64', 'float64']:
                    column = col
                    break
        
        column = column or "age"  # Fallback
        return f"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

plt.figure(figsize=(10, 6))
plt.hist(df['{column}'], bins=20, alpha=0.7, color='skyblue')
plt.title('Distribution of {column}')
plt.xlabel('{column}')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
result = df['{column}'].describe()
"""
    
    elif "correlation" in query:
        # Correlation analysis
        return """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Get numeric columns
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Calculate correlation matrix
corr_matrix = df[numeric_cols].corr()

# Create a heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
            linewidths=0.5, fmt='.2f')
plt.title('Correlation Matrix of Numeric Variables')
plt.tight_layout()

# Store the result
result = corr_matrix
"""
    
    elif "average" in query or "mean" in query:
        # Average/mean calculation
        group_by = None
        if "by" in query:
            # Try to extract grouping column
            parts = query.split("by")
            if len(parts) > 1 and df is not None:
                possible_col = parts[1].strip().split()[0]
                if possible_col in df.columns:
                    group_by = possible_col
        
        if group_by:
            return f"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Calculate average by group
result = df.groupby('{group_by}').mean()

# Plot the results
plt.figure(figsize=(10, 6))
result.plot(kind='bar')
plt.title('Average Values by {group_by}')
plt.ylabel('Average')
plt.xticks(rotation=45)
plt.tight_layout()
"""
        else:
            return """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Calculate means for numeric columns
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
result = df[numeric_cols].mean().sort_values(ascending=False)

# Plot the results
plt.figure(figsize=(10, 6))
result.plot(kind='bar')
plt.title('Average Values')
plt.ylabel('Average')
plt.xticks(rotation=45)
plt.tight_layout()
"""
    
    elif "sales" in query and "region" in query:
        # Sales by region
        return """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Calculate sales by region
result = df.groupby('region')['total'].sum().sort_values(ascending=False)

# Plot the results
plt.figure(figsize=(10, 6))
result.plot(kind='bar', color='skyblue')
plt.title('Sales by Region')
plt.ylabel('Total Sales')
plt.xlabel('Region')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
"""
    
    elif "outliers" in query:
        # Find outliers
        return """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Find numeric columns
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Use first numeric column found or 'total' as fallback
target_col = 'total' if 'total' in numeric_cols else numeric_cols[0] if numeric_cols else None

if target_col:
    # Calculate z-scores
    z_scores = stats.zscore(df[target_col])
    
    # Find outliers (|z| > 3)
    outliers = df[np.abs(z_scores) > 3]
    
    # Create boxplot
    plt.figure(figsize=(10, 6))
    plt.boxplot(df[target_col])
    plt.title(f'Boxplot of {target_col} with Outliers')
    plt.grid(True, alpha=0.3)
    
    result = outliers
else:
    result = "No numeric columns found for outlier detection"
"""
    
    else:
        # Default to a simple summary
        return """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Generate a basic summary of the dataframe
result = df.describe()

# Create a basic visualization
plt.figure(figsize=(10, 6))
if len(df.columns) > 0:
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if numeric_cols:
        df[numeric_cols[:5]].plot()
        plt.title('Overview of Key Metrics')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
"""