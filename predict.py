import joblib

# 1) Trained model load
model = joblib.load("chat_detector_model.pkl")

# 2) Predict function
def predict_chat(text):
    result = model.predict([text])[0]
    return result

# 3) Test examples
if __name__ == "__main__":
    tests = [
        "Hey where are you?",
        "Your bank account will be blocked today",
        "Forwarded many times please share",
        "Send OTP now to continue service"
    ]

    for t in tests:
        print(f"Chat: {t}")
        print("Prediction:", predict_chat(t))
        print("-" * 40)
