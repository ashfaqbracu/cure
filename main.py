from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
import time
import asyncio

load_dotenv()

# List of 5 example prompts
prompts = [
    "Explain to me how AI works",
    "What are the benefits of renewable energy?",
    "How does photosynthesis work in plants?",
    "Describe the process of machine learning",
    "What is the theory of relativity in simple terms?",
]


async def process_single_prompt(client, prompt, prompt_number):
    """Process a single prompt and return the result with timing."""
    print(f"--- Prompt {prompt_number}: {prompt} ---")

    start_time = time.time()

    response = await client.chat.completions.create(
        model="gemma-3-27b-it",
        messages=[
            {"role": "user", "content": prompt},
        ],
    )

    end_time = time.time()
    prompt_time = end_time - start_time

    result = {
        "prompt_number": prompt_number,
        "prompt": prompt,
        "response": response.choices[0].message.content,
        "time_taken": prompt_time,
    }

    print(f"Response: {response.choices[0].message.content}")
    print(f"Time taken: {prompt_time:.2f} seconds\n")

    return result


async def process_all_prompts():
    """Process all prompts concurrently using asyncio."""
    print("Processing 5 prompts concurrently...\n")
    total_start_time = time.time()

    # Create client with proper context management
    async with AsyncOpenAI(
        api_key=os.getenv("GOOGLE_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    ) as client:
        # Create tasks for all prompts to run concurrently
        tasks = [
            process_single_prompt(client, prompt, i + 1)
            for i, prompt in enumerate(prompts)
        ]

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks)

    total_end_time = time.time()
    total_time = total_end_time - total_start_time

    print("=== SUMMARY ===")
    print(f"Total prompts processed: {len(prompts)}")
    print(f"Total time taken: {total_time:.2f} seconds")
    print(f"Average time per prompt: {total_time / len(prompts):.2f} seconds")

    # Show individual timing details
    print("\n=== INDIVIDUAL TIMINGS ===")
    for result in results:
        print(f"Prompt {result['prompt_number']}: {result['time_taken']:.2f} seconds")

    return results


# Run the async function
async def main():
    """Main function with proper error handling."""
    try:
        await process_all_prompts()
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
