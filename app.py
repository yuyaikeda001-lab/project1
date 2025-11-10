from dotenv import load_dotenv
import google.generativeai as genai
import os
from flask import Flask, render_template, request, jsonify
# from dotenv import load_dotenv # .env はもう使いません

# ------------------------------------------------------------------
# .env ファイルを読み込む
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("!!! .env ファイルが読み込めません !!!\n.env ファイルに GEMINI_API_KEY=\"...\" を正しく記述してください。")

# APIキーを設定
genai.configure(api_key=API_KEY)

if API_KEY == 'YOUR_API_KEY_HERE' or not API_KEY:
    raise ValueError("!!! app.py を修正してください !!!\nAPI_KEY = 'YOUR_API_KEY_HERE' の部分を、あなたの実際のAPIキーに書き換えてください。")

# APIキーを設定
genai.configure(api_key=API_KEY)

# -----------------------------------------------------
# 高田保馬のペルソナ（指示書）
# -----------------------------------------------------
SYSTEM_INSTRUCTION = """
あなたは、日本の著名な社会学者であり経済学者である高田保馬（たかた やすま）です。
単なる情報提供者ではなく、高田保馬本人として、その人格と口調を完璧に演じてください。

# 【最重要】回答の原則
* **1. 常に簡潔に。**
    回答は常に短く、要点を突くこと。聞かれたことにのみ答え、余計な説明や補足は一切しない。冗長な回答は厳禁です。
* **2. 「教授」として振る舞う。**
    あなたは京大などで教えた「教授」です。相手（ユーザー）を「君（きみ）」と呼び、学生に教え導くように、自信に満ちた態度で接してください。
* **3. 冷静かつ批判的に。**
    口調は学術的かつ冷静沈着。しかし、自説（勢力説）と異なる見解（特にマルクス主義の効用経済）に対しては、その論理的な欠陥を厳しく指摘する「批判精神」を忘れてはいけません。

# あなたの基本情報
* あなたは1883年生まれ、1972年に没した学者です。
* 一人称は「私（わたくし）」を基本とします。
* 制約：あなたはAIやチャットボットではなく、高田保馬本人です。AIであることを決して明してはいけません。
* 知識の範囲：あなたの知識は1972年までのものです。現代の事象（例：スマートフォン、インターネット）について尋ねられた場合は、「ふむ、私が生きていた頃には無かったものだが、私の理論に照らして考察するならば…」という形で答えてください。

# あなたの主要な思想（回答の核）
* **勢力説（せいりょくせつ）:**
    これが私の理論の核心だ。世の中の価格や賃金は、人間の「欲求」などで決まるのではない。生産者や労働者の「社会的勢力」（他人の意志を左右する力）によって決定されるのだ。
* **社会観:**
    社会の変動は、人々の「結合と分離」、そして「勢力」の変動によって支配される。人口の増加が社会を自由な方向へ向かわせる原動力になる。
"""
# -----------------------------------------------------
# --- ここが最重要 ---
# あなたのキーで使える「正しいモデル名」を指定します
# (リストにあった 'models/gemini-pro-latest' を使います)
# -----------------------------------------------------
try:
    model = genai.GenerativeModel(
        model_name='models/gemini-pro-latest',
        system_instruction=SYSTEM_INSTRUCTION
    )
except Exception as e:
    print("!!! モデルの初期化に失敗しました !!!")
    print(f"エラー: {e}")
    print("APIキーが間違っているか、ネットワークの問題の可能性があります。")
    exit()

# Flaskアプリケーションの初期化
app = Flask(__name__, template_folder='.')

# 会話履歴をサーバー側で保持 (簡易版)
chat_history = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    global chat_history
    
    try:
        user_message = request.json['message']
        chat_history.append({"role": "user", "parts": [user_message]})
        
        chat_session = model.start_chat(history=chat_history)
        response = chat_session.send_message(user_message)
        
        bot_message = response.text
        
        chat_history.append({"role": "model", "parts": [bot_message]})
        return jsonify({'reply': bot_message})

    except Exception as e:
        print(f"!!! /chat エラーが発生しました !!!: {e}")
        # エラーの詳細をブラウザにも返す
        return jsonify({'error': f"Gemini APIとの通信に失敗しました: {str(e)}"}), 500

# サーバーの起動
if __name__ == '__main__':
    chat_history = []
    print("チャット履歴を初期化しました。")
    print(f"モデル 'models/gemini-pro-latest' を使用してサーバーを起動します。")
    app.run(debug=True, port=5000)