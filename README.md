# VoxAnalytica
# DataWhisperer

A voice-powered data analysis assistant that runs entirely on your local machine. DataWhisperer lets you ask questions about your data in natural language and get instant visualizations and insights.

![DataWhisperer Demo](https://via.placeholder.com/800x400?text=DataWhisperer+Demo)

## Features

- **🎙️ Voice Interface**: Ask questions about your data using natural speech
- **🧠 Local LLM Processing**: Powered by Phi-2 (2.7B parameter model)
- **📊 Instant Visualizations**: Automatically creates appropriate charts and graphs
- **🔒 Privacy-Focused**: All processing happens locally - no data leaves your machine
- **⚡ Mac M3 Optimized**: Specifically designed for efficiency on Apple Silicon

## How It Works

1. Speak or type your data question
2. DataWhisperer transcribes your speech using Whisper
3. A local LLM (Phi-2) converts your question to pandas code
4. The code executes against your dataset
5. Results and visualizations are displayed instantly

## Example Queries

- "Show me the distribution of customer ages"
- "What's our best-selling product in Q1?"
- "Calculate the correlation between customer spending and satisfaction"
- "Identify customers who haven't made a purchase in the last 90 days"
- "Plot monthly sales trends and highlight seasonal patterns"
- "Which customer segment has the highest lifetime value?"

## Installation

### Prerequisites

- Python 3.9+
- 8GB+ RAM (optimized for M-series Mac)
- Microphone for voice input

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/DataWhisperer.git
cd DataWhisperer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download models (creates models/ directory and downloads required models)
python download_models.py

# Run application
streamlit run app.py
```

## Technical Details

### Architecture

- **Speech Recognition**: Whisper.cpp (tiny model)
- **LLM**: Phi-2 (2.7B) quantized to 4-bit
- **Data Processing**: Pandas, Numpy, Matplotlib
- **UI**: Streamlit

### Memory Optimization

DataWhisperer is specifically designed to run on machines with limited RAM (8GB):

- **Model Quantization**: 4-bit quantization reduces model size by 75%
- **Efficient Inference**: llama.cpp for optimized local inference
- **Progressive Loading**: Components are loaded only when needed
- **Batch Processing**: Data is processed in manageable chunks

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
