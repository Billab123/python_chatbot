from bot_core import get_response

print("🤖 Chatbot is running! (type 'quit' to stop)\n")

while True:
    inp = input("You: ")
    if inp.lower() == "quit":
        break

    response = get_response(inp)
    print("Bot:", response)