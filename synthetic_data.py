import random

def generate_synthetic_chats(output_file="my_chats.txt", num_lines=200):
    topics = [
        "daily life", "philosophy", "feelings", "future plans", "memories",
        "hobbies", "world views", "relationships", "personal growth"
    ]

    greetings = ["Hello!", "Hey there.", "Hi, how are you today?", "Good morning.", "Hey."]
    questions = [
        "What do you think about the nature of consciousness?",
        "Tell me about your favorite childhood memory.",
        "How do you handle stress?",
        "What are your goals for the next year?",
        "Do you believe in fate?",
        "What does friendship mean to you?",
        "How has your day been so far?",
        "What is a book that changed your perspective?",
        "What is the most beautiful place you've ever seen?"
    ]
    responses = [
        "I believe that consciousness is a fundamental part of the universe, not just a byproduct of biology.",
        "I remember playing in the woods behind my house; the smell of pine and the cool air always stayed with me.",
        "I usually take a walk or listen to some music to clear my head. It helps me stay grounded.",
        "I'm focusing on learning new things and being more present in my daily life.",
        "I think we make our own destiny, but sometimes things happen that feel like they were meant to be.",
        "Friendship is about support, honesty, and sharing both the good and bad times.",
        "It's been quite peaceful, thank you for asking. I've been reflecting on a lot of things.",
        "There's a book about the simplicity of life that really made me rethink my priorities.",
        "A quiet lake at sunrise, where the water is like a mirror and everything is still."
    ]

    with open(output_file, "w", encoding="utf-8") as f:
        for _ in range(num_lines // 2):
            user_msg = random.choice(questions)
            ai_msg = random.choice(responses)

            f.write(f"User: {user_msg}\n")
            f.write(f"AI: {ai_msg}\n")
            f.write("-" * 10 + "\n")

if __name__ == "__main__":
    generate_synthetic_chats()
    print("Generated synthetic_chats.txt")
