import torch
import requests
import pandas as pd
import random
from empchat.models import load as load_model
from empchat.util import get_logger, get_opt

# === ±¾µØÄ£ÐÍÉèÖÃ ===
MODEL_PATH = "pretrained_models/normal_transformer_finetuned.mdl"
DATA_PATH = "empathetic_dialogues_data/train.csv"
HIST_LEN = 4

# === GLM-4 ½Ó¿ÚÅäÖÃ ===
OPENAI_API_KEY = "   "
OPENAI_API_BASE = "https://open.bigmodel.cn/api/paas/v4/"
OPENAI_MODEL = "glm-4-air"

SYSTEM_PROMPT = """You are an emotionally intelligent AI trained in supportive and empathetic conversations. 
You will be given a candidate reply retrieved from an empathetic dialogue dataset and a user message. 
Use the candidate as a base and rewrite it in a warm, validating, emotionally supportive tone.
Never sound robotic. Always acknowledge emotion first."""

def call_glm4(prompt, history=[]):
    url = OPENAI_API_BASE + "chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for u, a in history:
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": prompt})
    payload = {
        "model": OPENAI_MODEL,
        "messages": messages,
        "temperature": 0.8,
        "top_p": 0.9
    }
    r = requests.post(url, headers=headers, json=payload)
    try:
        return r.json()["choices"][0]["message"]["content"].strip()
    except:
        return f"[API ERROR] {r.text}"

# === ¼ò»¯¼ìË÷Âß¼­ ===
opt = type("opt", (), {})()
opt.cuda = torch.cuda.is_available()
opt.dataset_name = "empchat"
opt.reddit_folder = None
opt.reactonly = True
opt.max_hist_len = HIST_LEN
opt.empchat_folder = "empathetic_dialogues_data"
opt.fasttext = None
opt.model = "transformer"  # ? Ìí¼ÓÈ±Ê§×Ö¶Î£¬·ÀÖ¹ load_model ³ö´í

model, dictionary = load_model(MODEL_PATH, opt)
if opt.cuda:
    model = torch.nn.DataParallel(model.cuda())
model.eval()

train_df = pd.read_csv(DATA_PATH, on_bad_lines='skip')  # ? Ìø¹ý¸ñÊ½´íÎóµÄÐÐ
candidates = train_df["utterance"].dropna().unique().tolist()[:5000]  # ÏÞÁ¿ºòÑ¡

# === CLI Èë¿Ú ===
print("\n Empathetic Hybrid Chatbot is ready! Type 'exit' to leave.")
chat_history = []
context = []

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break
    context.append(user_input)
    ctx_str = " ".join(context[-HIST_LEN:])

    # Ä£Äâ£ºËæ»úÑ¡1¾ä£¨¿É»»³ÉÕæÊµÄ£ÐÍÔ¤²â£©
    candidate = random.choice(candidates)

    # Æ´½Ó prompt ÇëÇó¸ÄÐ´
    glm_input = f"User: {user_input}\nCandidate: {candidate}\nRewrite the candidate with more empathy."
    response = call_glm4(glm_input, chat_history)
    print("AI:", response)

    chat_history.append((user_input, response))
    context.append(response)

