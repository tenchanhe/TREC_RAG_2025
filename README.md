# TREC RAG 2025

This repository contains the implementation for the TREC 2025 RAG (Retrieval-Augmented Generation) Track. The project implements a hybrid retrieval system combined with advanced generation capabilities for the MS MARCO V2.1 corpus.

## 🎯 Project Overview

The project implements three main tracks:
- **Track R**: Retrieval Task
- **Track AG**: Augmented Generation Task
- **Track RAG**: Retrieval-Augmented Task

## 📋 Prerequisites

- Python 3.11+
- CUDA-capable GPU (recommended)
- 50GB+ free disk space
- 16GB+ RAM

## 🚀 Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/TREC_RAG_2025.git
   cd TREC_RAG_2025
   ```

2. Create and activate virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   # or
   .venv\Scripts\activate  # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the corpus:
   ```bash
   ./scripts/download.sh
   ```

5. Run the pipeline:
   ```bash
   ./scripts/run_pipeline.sh
   ```

## 📁 Project Structure

```
project_root/
├── data/                      # Data directory
├── indexes/                   # Search indices
├── runs/                      # Results
├── src/                       # Source code
├── configs/                   # Configuration files
├── tests/                     # Test files
├── logs/                      # Log files
├── notebooks/                 # Jupyter notebooks
├── docs/                      # Documentation
└── scripts/                   # Utility scripts
```

## ⚙️ Configuration

Configuration files are located in the `configs/` directory:
- `data_config.yaml`: Data processing settings
- `retrieval_config.yaml`: Retrieval system settings
- `generation_config.yaml`: Generation model settings

## 🧪 Usage

### Data Processing
```bash
python src/data/process_corpus.py --config configs/data_config.yaml
```

### Retrieval
```bash
python src/retrieval/run_retrieval.py --config configs/retrieval_config.yaml
```

### Generation
```bash
python src/generation/run_generation.py --config configs/generation_config.yaml
```

## 📊 Evaluation

Run evaluation:
```bash
python src/evaluation/evaluate.py --config configs/evaluation_config.yaml
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Contact

For questions and support, please open an issue in the GitHub repository.

## 🙏 Acknowledgments

- TREC RAG Track organizers
- MS MARCO team
- All contributors and participants
