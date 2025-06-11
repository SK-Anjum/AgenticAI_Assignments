from dotenv import load_dotenv
import os
from agents import Agent, Runner, RunConfig, OpenAIChatCompletionsModel, AsyncOpenAI 


load_dotenv()
gemini_api_key=os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

agent = Agent(
    name="Smart Assistant", 
    instructions="You are a helpful assistant that answers questions.")

result1 = Runner.run_sync(
    agent,
    input= "Answer who discovered garvity",
    run_config=config)


result2 = Runner.run_sync(
    agent,
    input= "Provide two study tips for students",
    run_config=config)

result3 = Runner.run_sync(
    agent,
    input= "Summarize the given text in 2 lines: 'The Amazon Rainforest, often called the lungs of the Earth produces around 20 of the world's oxygen. It spans nine countries in South America, with the largest portion in Brazil. This vast rainforest is home to millions of species of plants, animals, and insects. It plays a vital role in regulating the global climate by absorbing carbon dioxide. However, deforestation caused by logging, farming, and mining threatens its delicate ecosystem. Protecting the Amazon is essential for maintaining biodiversity and fighting climate change.'",
    run_config=config)

print(result1.final_output)
print(result2.final_output)
print(result3.final_output)