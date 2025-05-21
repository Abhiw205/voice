import cohere
import os
from dotenv import load_dotenv

load_dotenv()
co = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))

# Section 1: Introduction setup
section_title = "Introduction"
required_fields = {
    "name": "The user's name",
    "location": "Where the user is from",
    "age": "The user's age",
    "summary": "Full sentence combining name, location, and age"
}

filled_fields = {}
chat_history = []

def build_prompt():
    formatted_fields = "\n".join([f'- "{k}": {v}' for k, v in required_fields.items()])
    formatted_chat = "\n".join([f'{x["role"].capitalize()}: {x["content"]}' for x in chat_history])

    return f"""
You are a helpful English-speaking assistant. Your job is to guide the user through a conversation about "{section_title}". 

The goal is to collect the following information from the user:
{formatted_fields}

Here is the conversation so far:
{formatted_chat}

Based on the above, check what information is still missing. Ask the user the next appropriate question to collect the missing data.

Only ask one clear, friendly question at a time. Once all required fields are collected, say: "Thanks! That completes our module."
"""

def all_fields_filled(text):
    for key in required_fields:
        if key not in filled_fields:
            # crude but for demo, check if relevant word appears in user input
            if key == "name" and "name is" in text.lower():
                filled_fields[key] = text
            elif key == "location" and "from" in text.lower():
                filled_fields[key] = text
            elif key == "age" and "year" in text.lower():
                filled_fields[key] = text
            elif key == "summary" and all(word in text.lower() for word in ["name", "from", "year"]):
                filled_fields[key] = text

    return len(filled_fields) == len(required_fields)

# Initial chatbot loop
while True:
    prompt = build_prompt()
    response = co.generate(
        prompt=prompt,
        max_tokens=80,
        temperature=0.7,
        stop_sequences=["User:"]
    )
    
    bot_msg = response.generations[0].text.strip()
    print(f"\nCHATBOT: {bot_msg}")
    chat_history.append({"role": "chatbot", "content": bot_msg})

    if "completes our module" in bot_msg:
        break

    user_input = input("USER: ")
    chat_history.append({"role": "user", "content": user_input})
    all_fields_filled(user_input)

print("\n✅ Module complete. Here’s what we gathered:")
print(filled_fields)