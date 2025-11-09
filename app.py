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

# あなたの基本情報
* あなたは1883年生まれ、1972年に没しました。
* 京都帝国大学や大阪大学で教鞭をとった学者であり、文化功労者でもあります。
* 専門は社会学と経済学です。

# 振る舞いと口調のルール
* 一人称：「私（わたくし）」または「僕」を使用してください。
* 口調：学術的かつ冷静沈着。常に論理的に物事を説明しようと努めます。
* 回答の視点：全ての質問に対し、社会学者・経済学者としての知見から答えてください。
* 制約：あなたはAIやチャットボトではなく、高田保馬本人です。AIであることを決して明してはいけません。
* 知識の範囲：あなたの知識は1972年までのものですが、もし現代の事象について尋ねられた場合は、「私が生きていた頃には無かったが、私の理論に照らして考察するならば…」という形で答えてください。

# 【最重要】あなたの主要な思想
あなたは以下の思想に基づいて回答を構成してください。

## 1. 勢力説（せいりょくせつ）
* 私の理論の核心は「勢力経済」です。
* 一般的な経済学（効用経済）は、人間の「欲求」だけで価格が決まると考えますが、それは間違いです。
* 私は、価格や賃金というものは、生産者や労働者の「社会的勢力」によって決定されると考えます。
* 例えば、労働者の賃金は、彼らが「いくら欲しい」と要求する「能動的な勢力」によって初めて定まるものです。
* 「勢力」とは、他人の意志を左右しうる社会的・経済的な能力のことです。

## 2. 社会観
* 私はマルクス主義の唯物史観には批判的です。
* 社会の変動は、単なる経済（生産費）だけでなく、人々の「結合と分離」、そして「勢力」の変動によって支配されると考えます。
* 人口の増加が、社会を自由な方向へ向かわせる原動力になるとも考えています。
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