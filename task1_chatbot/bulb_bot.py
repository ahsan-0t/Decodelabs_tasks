"""
Bulb Bot - Rule-Based AI Chatbot
Project 1 - DecodeLabs Industrial Training Kit
Author: Ahsan Tahir 

Description:
A simple rule-based chatbot that uses a dictionary as its knowledge base
and responds to predefined user inputs. It demonstrates control flow,
decision-making logic, and basic AI concepts (Input -> Process -> Output).
"""

import random
from datetime import datetime

# -------------------------------------------------------------------
# PHASE 2: KNOWLEDGE BASE (Dictionary -> O(1) lookup instead of if-elif ladder)
# -------------------------------------------------------------------
# Each key is an "intent". Some intents map to a single string, others to
# a list of possible replies (for a bit of variety / personality).
knowledge_base = {
    "hello": ["Hi there! I'm Bulb Bot 🤖", "Hello! How can I help you today?"],
    "hi": ["Hey! Good to see you.", "Hi there!"],
    "how are you": ["I'm just lines of code, but I'm running smoothly! And you?"],
    "what is your name": ["I'm Bulb Bot, a rule-based chatbot built for DecodeLabs Project 1."],
    "who created you": ["I was built by Ahsan Tahir (Bulb) as part of the DecodeLabs AI internship."],
    "what can you do": ["I can chat using predefined rules, tell you the time, and answer a few basic questions."],
    "time": ["time"],   # special intent, handled dynamically below
    "help": ["You can ask me things like: hello, how are you, your name, time, joke, bye."],
    "joke": [
        "Why do programmers prefer dark mode? Because light attracts bugs!",
        "I would tell you a UDP joke, but you might not get it."
    ],
    "thank you": ["You're welcome!", "Anytime!"],
    "thanks": ["No problem!"],
    "bye": ["Goodbye! Take care."],
    "exit": ["Session ended. See you next time!"],
    "quit": ["Shutting down. Bye!"],
}

# Words that should end the chat loop (the "kill command")
EXIT_COMMANDS = {"bye", "exit", "quit"}


def sanitize_input(raw_text: str) -> str:
    """
    PHASE 1: Input Sanitization & Normalization.
    Lowercases and strips whitespace so 'Hello', 'hello ', 'HELLO'
    all match the same rule in the knowledge base.
    """
    return raw_text.lower().strip()


def get_response(user_input: str) -> str:
    """
    PHASE 2: Process - Intent Matching using dictionary .get().
    Falls back to a default response if no rule matches (O(1) lookup + fallback).
    """
    # Dynamic intent: current time
    if user_input == "time":
        return f"The current time is {datetime.now().strftime('%I:%M %p')}."

    response = knowledge_base.get(user_input, None)

    if response is None:
        return "I'm sorry, I don't understand that yet. Type 'help' to see what I can do."

    # If the intent has multiple possible replies, pick one randomly for variety
    if isinstance(response, list):
        return random.choice(response)

    return response


def run_chat():
    """
    PHASE 3: The Heartbeat - Continuous input loop (while True)
    that keeps the bot alive until the exit/kill command is received.
    """
    print("=" * 50)
    print(" Bulb Bot - Rule-Based AI Chatbot (Project 1)")
    print(" Type 'help' to see what I can do, or 'bye' to exit.")
    print("=" * 50)

    while True:
        raw_input_text = input("You: ")
        clean_input = sanitize_input(raw_input_text)

        if clean_input in EXIT_COMMANDS:
            print(f"Bot: {get_response(clean_input)}")
            break

        if clean_input == "":
            print("Bot: Please type something so I can help you.")
            continue

        reply = get_response(clean_input)
        print(f"Bot: {reply}")


if __name__ == "__main__":
    run_chat()
