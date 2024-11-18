from transformers import pipeline

messages = [
    {"role": "user", "content": "Who are you?"},
]
while True:
    inp = input("Enter your message: ")
    messages.append({"role": "user", "content": inp})
    pipe = pipeline("text-generation", model="microsoft/DialoGPT-medium", max_tokens=500,)
    print(pipe(messages))
    # print(pipe(messages[0]["generated_text"])[1])