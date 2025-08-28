# CureGemma - Medical AI Evaluation Tool

A Python tool for evaluating medical AI models using the CureBench dataset. This project uses Google's Gemma-3-27B-IT model to process medical questions and generate comprehensive answers with reasoning traces.

## Overview

CureGemma processes medical questions from the CureBench testset Phase 1, which contains 2,080 medical questions across different question types:
- Multiple choice questions
- Open-ended multiple choice questions  
- Open-ended questions

The tool generates predictions, reasoning traces, and formatted outputs suitable for medical AI evaluation frameworks.

## Features

- **Async Processing**: Concurrent processing of multiple questions with proper rate limiting
- **Retry Logic**: Robust error handling with exponential backoff using Tenacity
- **Multiple Question Types**: Support for different medical question formats
- **Rate Limiting**: Built-in rate limiting (15 requests per minute) to respect API limits
- **Comprehensive Output**: Generates CSV submissions, metadata, and ZIP files for evaluation
- **Progress Tracking**: Real-time progress monitoring and intermediate saves

## Installation

1. Clone this repository:
```bash
git clone https://github.com/ashfaqbracu/cure.git
cd cure
```

2. Install dependencies using uv (recommended) or pip:
```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

3. Set up your environment variables by creating a `.env` file:
```bash
GOOGLE_API_KEY=your_google_api_key_here
```

## Usage

### Processing CureBench Dataset

Run the main evaluation script:
```bash
python cure.py
```

This will:
- Load the CureBench testset Phase 1 data
- Process all 2,080 questions with rate limiting
- Generate comprehensive answers and reasoning traces
- Save results in multiple formats (CSV, JSON, ZIP)

### Testing the API Connection

Run the example script to test your API connection:
```bash
python main.py
```

This will process 5 example prompts concurrently to verify your setup.

## Project Structure

```
cure/
├── cure.py                           # Main evaluation script
├── main.py                          # API connection test script
├── curebench_testset_phase1.jsonl   # CureBench dataset
├── pyproject.toml                   # Project configuration
├── uv.lock                          # Dependency lock file
├── .env                             # Environment variables (create this)
├── .gitignore                       # Git ignore rules
└── README.md                        # This file
```

## Output Format

The tool generates three types of output files in the `submission/` directory:

### CSV File
Contains the evaluation results with columns:
- `id`: Question identifier
- `prediction`: Model's answer/prediction
- `choice`: Selected choice (A/B/C/D/E) for multiple choice questions
- `reasoning`: Detailed reasoning trace

### Metadata JSON
Contains processing statistics and model information:
- Model details (gemma-3-27b-it)
- Processing statistics
- Timestamps and configuration

### ZIP File
Packaged submission containing both CSV and metadata files.

## Rate Limiting

The tool implements careful rate limiting:
- **15 requests per minute** to stay within API limits
- **Batch processing** of 15 questions per batch
- **Automatic waiting** between batches
- **Progress tracking** with time estimates

## Error Handling

Robust error handling includes:
- **Retry logic** with exponential backoff
- **Connection error recovery**
- **Rate limit handling**
- **Graceful degradation** for failed requests
- **Intermediate saves** to prevent data loss

## Question Type Processing

### Multiple Choice
- Extracts single letter answers (A/B/C/D/E)
- Uses concise prompting for efficiency

### Open-ended Multiple Choice
- Generates detailed reasoning
- Extracts final choice from comprehensive analysis
- Follows structured format: REASONING + ANSWER

### Open-ended
- Provides comprehensive medical explanations
- No choice extraction required

## Dependencies

- `openai>=1.101.0` - API client for Gemma model
- `python-dotenv>=1.0.1` - Environment variable management
- `tenacity>=9.0.0` - Retry logic with exponential backoff

## Configuration

Key configuration options in `cure.py`:
- `temperature=0.1` - Low temperature for consistent medical answers
- `max_tokens=1000` - Response length limit
- `batch_size=15` - Questions per batch for rate limiting

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source. Please check the repository for license details.

## Acknowledgments

- CureBench dataset for medical AI evaluation
- Google's Gemma model for medical reasoning
- OpenAI-compatible API interface

## Support

For issues or questions:
1. Check the error messages and logs
2. Verify your API key and rate limits
3. Review the CureBench dataset format
4. Open an issue on GitHub if needed