from dotenv import load_dotenv
import os
from agents import Agent, Runner, RunConfig, OpenAIChatCompletionsModel, AsyncOpenAI 

# Load environment variables
load_dotenv()

# Load API Key 
gemini_api_key=os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

# Initialize the external OpenAI-compatible client
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Set up the model
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

# Configure run settings
config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

# Create the agent
agent = Agent(
    name="Smart Assistant",
    instructions=(
        "You are a helpful assistant that can:\n"
        "- Answer academic questions\n"
        "- Provide study tips\n"
        "- Summarize small text passages"
    )
)

# CLI menu function
def main():
    print("🎓 Welcome to the Smart Assistant!")
    
    while True:
        print("\nPlease choose an option:")
        print("1. Answer an academic question")
        print("2. Get study tips")
        print("3. Summarize a short text passage")
        print("4. Exit")

        choice = input("Enter your choice (1-4): ").strip()

        if choice == "1":
            question = input("Enter your academic question: ")
            prompt = f"Answer the following academic question: {question}"

        elif choice == "2":
            topic = input("Enter the study topic (e.g., exams, focus, math): ")
            prompt = f"Provide two study tips for students studying {topic}"

        elif choice == "3":
            passage = input("Enter a short text passage to summarize: ")
            prompt = f"Summarize the following text in 2 lines: '{passage}'"

        elif choice == "4":
            print("👋 Exiting. Good luck with your studies!")
            break

        else:
            print("Invalid choice. Please select from 1 to 4.")
            continue

        try:
            result = Runner.run_sync(agent, input=prompt, run_config=config)
            print("\n🧠 Assistant's Response:")
            print(result.final_output)
        except Exception as e:
            print(f"❌ Error: {e}")

# Run the main menu
if __name__ == "__main__":
    main()