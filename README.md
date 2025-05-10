# VoxAnalytica

Project Structurture:

VoxAnalytica/
├── README.md
├── app.py                  # Main application entry point
├── requirements.txt        # Dependencies
├── models/                 # Model storage
│   ├── whisper-tiny.en.pt  # Speech recognition model
│   └── phi-2-q4.gguf       # Quantized LLM
├── src/
│   ├── speech.py           # Speech handling
│   ├── llm.py              # LLM interface
│   ├── data_ops.py         # Data operations
│   ├── prompts.py          # System prompts
│   └── utils.py            # Utilities
├── examples/               # Example datasets
│   ├── sales.csv
│   └── customers.csv
└── ui/                     # Streamlit UI components
    ├── components.py
    └── styles.css