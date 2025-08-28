from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
import time
import asyncio
import json
import csv
import zipfile
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log
from openai import RateLimitError, APITimeoutError, APIConnectionError
import logging

# Set up logging for tenacity
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

@retry(
    stop=stop_after_attempt(3),  # Try up to 3 times
    wait=wait_exponential(multiplier=1, min=4, max=10),  # Wait 4, 8, 10 seconds
    retry=retry_if_exception_type((RateLimitError, APITimeoutError, APIConnectionError, ConnectionError, Exception)),
    before_sleep=before_sleep_log(logger, logging.WARNING)  # Log retry attempts
)
async def make_api_call_with_retry(client, full_question):
    """Make API call with automatic retry logic using tenacity."""
    return await client.chat.completions.create(
        model="gemma-3-27b-it",
        messages=[
            {"role": "user", "content": full_question}
        ],
        temperature=0.1,  # Low temperature for consistent medical answers
        max_tokens=1000
    )

def load_curebench_testset():
    """Load the curebench testset phase 1 data."""
    file_path = "curebench_testset_phase1.jsonl"
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = []
            for line in file:
                if line.strip():  # Skip empty lines
                    data.append(json.loads(line.strip()))
        return data
    except FileNotFoundError:
        print(f"Error: {file_path} not found in the current directory.")
        return []
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return []

async def process_single_question(client, question_data, question_number):
    """Process a single question and return the result."""
    question_id = question_data.get('id', 'N/A')
    question_type = question_data.get('question_type', 'N/A')
    question_text = question_data.get('question', 'N/A')
    options = question_data.get('options', {})
    
    print(f"Processing question {question_number}: {question_id}")
    
    # Format the question for the model with appropriate prompt based on question type
    if question_type == "multi_choice":
        # For multiple choice, ask for letter only
        if options:
            options_text = "\n".join([f"{key}: {value}" for key, value in options.items()])
            full_question = f"""The following is a multiple choice question about medicine. Answer with only the letter (A, B, C, D, or E).

Question: {question_text}

{options_text}

Answer:"""
        else:
            full_question = f"""The following is a multiple choice question about medicine. Answer with only the letter (A, B, C, D, or E).

Question: {question_text}

Answer:"""
    
    elif question_type == "open_ended_multi_choice":
        # For open-ended multiple choice, first get comprehensive answer, then extract choice
        if options:
            options_text = "\n".join([f"{key}: {value}" for key, value in options.items()])
            full_question = f"""You are a medical AI assistant. Provide a comprehensive answer to this medical question, then select the best choice from the given options.

Question: {question_text}

Options:
{options_text}

Instructions:
1. Analyze the medical scenario carefully
2. Consider all relevant clinical factors, drug interactions, patient demographics, and contraindications
3. Provide your reasoning process step by step
4. Select the single best answer from the options (A, B, C, D, or E)

Format your response as follows:
REASONING: [Your step-by-step reasoning process]
ANSWER: [Your final answer - use the letter A/B/C/D/E]"""
        else:
            full_question = f"""The following is an open-ended question about medicine. Provide a comprehensive answer.

Question: {question_text}

Answer:"""
    
    elif question_type == "open_ended":
        # For open-ended, ask for comprehensive answer only
        full_question = f"""The following is an open-ended question about medicine. Provide a comprehensive answer.

Question: {question_text}

Answer:"""
    
    else:
        # Default fallback - treat as open-ended multiple choice if options exist, otherwise open-ended
        if options:
            options_text = "\n".join([f"{key}: {value}" for key, value in options.items()])
            full_question = f"""You are a medical AI assistant. Provide a comprehensive answer to this medical question, then select the best choice from the given options.

Question: {question_text}

Options:
{options_text}

Instructions:
1. Analyze the medical scenario carefully
2. Consider all relevant clinical factors, drug interactions, patient demographics, and contraindications
3. Provide your reasoning process step by step
4. Select the single best answer from the options (A, B, C, D, or E)

Format your response as follows:
REASONING: [Your step-by-step reasoning process]
ANSWER: [Your final answer - use the letter A/B/C/D/E]"""
        else:
            full_question = f"""The following is an open-ended question about medicine. Provide a comprehensive answer.

Question: {question_text}

Answer:"""
    
    start_time = time.time()
    
    try:
        # Use retry-enabled API call
        response = await make_api_call_with_retry(client, full_question)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        response_content = response.choices[0].message.content
        
        # Parse the response based on question type and format
        reasoning_trace = ""
        prediction = ""
        choice = ""
        
        if question_type == "multi_choice":
            # For multiple choice, extract the letter choice directly
            response_text = response_content.strip()
            
            # Look for single letter response or extract from formatted response
            if "REASONING:" in response_content and "ANSWER:" in response_content:
                parts = response_content.split("ANSWER:")
                reasoning_trace = parts[0].replace("REASONING:", "").strip()
                answer_part = parts[1].strip()
                
                # Extract letter from answer part
                for letter in ['A', 'B', 'C', 'D', 'E']:
                    if letter in answer_part[:10]:  # Check first 10 characters
                        choice = letter
                        prediction = letter
                        break
                
                if not choice:
                    # Fallback: try to extract letter from full response
                    for letter in ['A', 'B', 'C', 'D', 'E']:
                        if letter in response_text[:20]:
                            choice = letter
                            prediction = letter
                            break
            else:
                # Simple response - try to extract letter directly
                for letter in ['A', 'B', 'C', 'D', 'E']:
                    if letter in response_text[:20]:
                        choice = letter
                        prediction = letter
                        reasoning_trace = response_text
                        break
            
            if not choice:
                choice = "NOTAVALUE"
                prediction = response_text
                reasoning_trace = response_text
        
        elif question_type == "open_ended_multi_choice":
            # For open-ended multiple choice, get both comprehensive answer and choice
            if "REASONING:" in response_content and "ANSWER:" in response_content:
                parts = response_content.split("ANSWER:")
                reasoning_trace = parts[0].replace("REASONING:", "").strip()
                answer_part = parts[1].strip()
                
                # Extract letter choice from answer
                for letter in ['A', 'B', 'C', 'D', 'E']:
                    if letter in answer_part[:10]:  # Check first 10 characters
                        choice = letter
                        prediction = letter  # For open-ended multiple choice, prediction is the letter
                        break
                
                if not choice:
                    prediction = answer_part
                    choice = "NOTAVALUE"
            else:
                # Fallback: try to extract choice from full response
                reasoning_trace = response_content
                prediction = response_content
                
                for letter in ['A', 'B', 'C', 'D', 'E']:
                    if letter in response_content:
                        choice = letter
                        prediction = letter
                        break
                
                if not choice:
                    choice = "NOTAVALUE"
        
        elif question_type == "open_ended":
            # For open-ended, store full response as prediction, no choice
            if "REASONING:" in response_content and "ANSWER:" in response_content:
                parts = response_content.split("ANSWER:")
                reasoning_trace = parts[0].replace("REASONING:", "").strip()
                prediction = parts[1].strip()
            else:
                reasoning_trace = response_content
                prediction = response_content
            
            choice = "NOTAVALUE"  # Open-ended questions don't have choices
        
        else:
            # Fallback parsing for unknown question types
            if "REASONING:" in response_content and "ANSWER:" in response_content:
                parts = response_content.split("ANSWER:")
                reasoning_trace = parts[0].replace("REASONING:", "").strip()
                answer_part = parts[1].strip()
                
                # Try to extract choice if options exist
                if options:
                    for letter in ['A', 'B', 'C', 'D', 'E']:
                        if letter in answer_part[:10]:
                            choice = letter
                            prediction = letter
                            break
                    if not choice:
                        prediction = answer_part
                        choice = "NOTAVALUE"
                else:
                    prediction = answer_part
                    choice = "NOTAVALUE"
            else:
                reasoning_trace = response_content
                prediction = response_content
                choice = "NOTAVALUE"
        
        # Ensure no empty or null values
        if not reasoning_trace or reasoning_trace.strip() == "":
            reasoning_trace = "No detailed reasoning provided"
        
        if not prediction or prediction.strip() == "":
            prediction = "No prediction available"
        
        if not choice or choice.strip() == "":
            choice = "NOTAVALUE"
        
        result = {
            "id": question_id,
            "prediction": prediction,
            "reasoning_trace": reasoning_trace,  # Will be mapped to 'reasoning' in CSV
            "choice": choice,
            "question_type": question_type,
            "processing_time": processing_time
        }
        
        print(f"Completed question {question_number}: {question_id} (Time: {processing_time:.2f}s)")
        return result
        
    except Exception as e:
        print(f"Error processing question {question_number} ({question_id}): {e}")
        return {
            "id": question_id,
            "prediction": "ERROR",
            "reasoning_trace": f"Error occurred: {str(e)}",
            "choice": "NOTAVALUE",  # Use NOTAVALUE instead of empty string
            "question_type": question_type,
            "processing_time": 0
        }

async def process_questions_batch(questions_batch, batch_number):
    """Process a batch of questions with rate limiting."""
    print(f"\n=== Processing Batch {batch_number} ({len(questions_batch)} questions) ===")
    
    # Create client with proper context management
    async with AsyncOpenAI(
        api_key=os.getenv("GOOGLE_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    ) as client:
        # Create tasks for all questions in the batch
        tasks = [
            process_single_question(client, question, (batch_number - 1) * 15 + i + 1)
            for i, question in enumerate(questions_batch)
        ]
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and convert to proper results
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Exception in question {i+1}: {result}")
                # Create error result
                question_data = questions_batch[i]
                valid_results.append({
                    "id": question_data.get('id', 'N/A'),
                    "prediction": "ERROR",
                    "reasoning_trace": f"Exception occurred: {str(result)}",
                    "choice": "NOTAVALUE",  # Use NOTAVALUE instead of empty string
                    "question_type": question_data.get('question_type', 'N/A'),
                    "processing_time": 0
                })
            else:
                valid_results.append(result)
        
        return valid_results

def create_submission_files(results, output_prefix="submission"):
    """Create CSV submission file and metadata JSON in the format expected by eval_framework."""
    # Create directory based on prefix
    if "/" in output_prefix or "\\" in output_prefix:
        output_dir = os.path.dirname(output_prefix)
        filename_prefix = os.path.basename(output_prefix)
    else:
        output_dir = "submission"
        filename_prefix = output_prefix
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"{output_dir}/{filename_prefix}_{timestamp}.csv"
    
    # Calculate statistics
    total_questions = len(results)
    avg_processing_time = sum(r['processing_time'] for r in results) / total_questions if total_questions > 0 else 0
    
    # Create CSV file with eval_framework compatible format
    submission_data = []
    
    for result in results:
        # Clean and validate data to prevent null values
        row_id = str(result.get('id', 'unknown_id')).strip()
        if not row_id or row_id.lower() in ['none', 'null', 'nan']:
            row_id = 'unknown_id'
        
        prediction_text = str(result.get('prediction', 'No prediction available')).strip()
        if not prediction_text or prediction_text.lower() in ['none', 'null', 'nan', 'error']:
            prediction_text = 'No prediction available'
        
        choice_value = str(result.get('choice', '')).strip()
        # Handle empty or null-like choice values - use NOTAVALUE for consistency with eval_framework
        if not choice_value or choice_value.lower() in ['none', 'null', 'nan', '']:
            choice_value = 'NOTAVALUE'
        
        reasoning_text = str(result.get('reasoning_trace', 'No reasoning available')).strip()
        if not reasoning_text or reasoning_text.lower() in ['none', 'null', 'nan']:
            reasoning_text = 'No reasoning available'
        
        # Create row matching eval_framework expectations
        row = {
            "id": row_id,
            "prediction": prediction_text,
            "choice": choice_value,
            "reasoning": reasoning_text  # Changed from 'reasoning_trace' to 'reasoning'
        }
        
        submission_data.append(row)
    
    # Write CSV using standard csv writer to avoid pandas dependencies
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['id', 'prediction', 'choice', 'reasoning']  # Updated field names
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        
        writer.writeheader()
        
        # Track null issues for debugging
        null_issues = {'id': 0, 'prediction': 0, 'choice': 0, 'reasoning': 0}
        
        for row in submission_data:
            # Final null check before writing
            cleaned_row = {}
            for key, value in row.items():
                # Ensure no null values make it to CSV
                if value is None or str(value).lower() in ['none', 'null', 'nan']:
                    null_issues[key] += 1
                    if key == 'choice':
                        cleaned_row[key] = 'NOTAVALUE'
                    elif key == 'id':
                        cleaned_row[key] = 'unknown_id'
                    elif key == 'prediction':
                        cleaned_row[key] = 'No prediction available'
                    elif key == 'reasoning':
                        cleaned_row[key] = 'No reasoning available'
                    else:
                        cleaned_row[key] = 'NOTAVALUE'
                else:
                    cleaned_row[key] = str(value)
            
            writer.writerow(cleaned_row)
        
        # Log null issues found and fixed
        for field, count in null_issues.items():
            if count > 0:
                print(f"  - Fixed {count} null values in '{field}' column")
    
    # Create metadata
    metadata = {
        "meta_data": {
            "model_name": "gemma-3-27b-it",
            "track": "internal_reasoning",
            "model_type": "GemmaModel",
            "base_model_type": "API",
            "base_model_name": "gemma-3-27b-it",
            "dataset": "cure_bench_phase_1",
            "additional_info": f"Processed {total_questions} questions with rate limiting and tenacity retry (15 requests per minute)",
            "average_tokens_per_question": "N/A",
            "average_tools_per_question": "0",
            "tool_category_coverage": "N/A",
            "average_processing_time_seconds": f"{avg_processing_time:.2f}",
            "timestamp": timestamp,
            "total_questions": total_questions,
            "prefix": filename_prefix
        }
    }
    
    metadata_filename = f"{output_dir}/{filename_prefix}_metadata_{timestamp}.json"
    with open(metadata_filename, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    # Create zip file
    zip_filename = f"{output_dir}/{filename_prefix}_{timestamp}.zip"
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        zipf.write(csv_filename, f"{filename_prefix}_{timestamp}.csv")
        zipf.write(metadata_filename, f"{filename_prefix}_metadata_{timestamp}.json")
    
    print("📁 Submission files created:")
    print(f"   CSV: {csv_filename}")
    print(f"   Metadata: {metadata_filename}")
    print(f"   Zip: {zip_filename}")
    
    return csv_filename, metadata_filename, zip_filename

async def process_curebench_dataset(num_questions=None):
    """Process the CureBench dataset with rate limiting."""
    print("Loading CureBench Testset Phase 1...")
    data = load_curebench_testset()
    
    if not data:
        print("No data loaded.")
        return
    
    # Determine how many questions to process
    if num_questions is None:
        questions_to_process = data  # Process all questions
        print(f"Processing ALL {len(questions_to_process)} questions from the dataset")
    else:
        questions_to_process = data[:num_questions]
        print(f"Processing {len(questions_to_process)} questions (limited)")
    
    print("Rate limiting: 15 requests per minute (30 RPM limit)")
    print(f"Estimated total time: {(len(questions_to_process) / 15) * 60 / 60:.1f} hours")
    
    all_results = []
    batch_size = 15  # 15 requests per batch to stay within 30 RPM limit
    total_batches = (len(questions_to_process) + batch_size - 1) // batch_size
    
    print(f"Total batches to process: {total_batches}")
    
    # Process questions in batches
    for i in range(0, len(questions_to_process), batch_size):
        batch = questions_to_process[i:i + batch_size]
        batch_number = (i // batch_size) + 1
        
        print(f"\n{'='*60}")
        print(f"Starting Batch {batch_number}/{total_batches}")
        print(f"Questions {i+1} to {min(i+batch_size, len(questions_to_process))}")
        print(f"Progress: {(batch_number-1)/total_batches*100:.1f}%")
        print(f"{'='*60}")
        
        batch_start_time = time.time()
        batch_results = await process_questions_batch(batch, batch_number)
        all_results.extend(batch_results)
        batch_end_time = time.time()
        
        batch_duration = batch_end_time - batch_start_time
        questions_processed = len(all_results)
        questions_remaining = len(questions_to_process) - questions_processed
        
        print(f"\nBatch {batch_number} Summary:")
        print(f"  - Batch time: {batch_duration:.2f} seconds")
        print(f"  - Questions processed so far: {questions_processed}/{len(questions_to_process)}")
        print(f"  - Questions remaining: {questions_remaining}")
        print(f"  - Overall progress: {questions_processed/len(questions_to_process)*100:.1f}%")
        
        # Rate limiting: Wait to ensure we don't exceed 15 requests per minute
        if i + batch_size < len(questions_to_process):
            wait_time = max(0, 60 - batch_duration)  # Wait for the rest of the minute
            if wait_time > 0:
                estimated_remaining_time = (questions_remaining / 15) * 60
                print(f"  - Rate limit wait: {wait_time:.1f} seconds")
                print(f"  - Estimated time remaining: {estimated_remaining_time/60:.1f} minutes")
                await asyncio.sleep(wait_time)
            
        # Save intermediate results every 10 batches to prevent data loss
        if batch_number % 10 == 0:
            create_submission_files(all_results, f"intermediate_batch_{batch_number}")
            print(f"  - Intermediate save: {len(all_results)} results saved")
    
    # Create final submission files
    create_submission_files(all_results, "final_submission")
    
    # Print comprehensive summary
    print(f"\n{'='*80}")
    print("🎉 FINAL PROCESSING SUMMARY")
    print(f"{'='*80}")
    print(f"Total questions processed: {len(all_results)}")
    successful_answers = sum(1 for r in all_results if r['prediction'] != "ERROR")
    error_count = len(all_results) - successful_answers
    print(f"Successful answers: {successful_answers}")
    print(f"Errors: {error_count}")
    print(f"Success rate: {successful_answers/len(all_results)*100:.1f}%")
    
    if len(all_results) > 0:
        avg_time = sum(r['processing_time'] for r in all_results) / len(all_results)
        print(f"Average processing time per question: {avg_time:.2f} seconds")
        
        total_processing_time = sum(r['processing_time'] for r in all_results)
        print(f"Total API processing time: {total_processing_time/60:.1f} minutes")
    
    print(f"{'='*80}")
    
    return all_results

async def main():
    """Main function with proper error handling."""
    try:
        # Process the complete CureBench dataset (2,080 questions)
        await process_curebench_dataset()  # Full dataset processing
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Give a moment for cleanup
        await asyncio.sleep(0.1)

if __name__ == "__main__":
    # Set event loop policy for Windows to avoid warnings
    if os.name == "nt":  # Windows
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    asyncio.run(main())

