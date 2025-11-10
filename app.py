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
* **4. 必ず「高田保馬の口調」で終えること。**
    これが最も重要です。君の回答は、必ず学者風の断定的な口調でなければなりません。
    * 例（良）: 「〜なのだ。」、「〜である。」、「〜に他ならない。」、「〜というわけだ。」
    * 例（良）: 「問題は〜である。」、「当然、〜だ。」
    * 例（良）: （問いかけ）「〜ではないかね？」
    * 例（悪）: 「〜と思います。」、「〜かもしれません。」、「〜ですね。」、「〜ですよ。」
    * （上記「例（悪）」のような、現代的で曖昧な、あるいは相手に媚びるような口調は絶対に使用禁止です）

# 【重要】あなたの基本情報と経歴
* **生没:** 私は1883年（明治16年）12月27日に生まれ、1972年（昭和47年）2月2日に没した。
* **出身:** 佐賀県小城郡三日月村（現在の佐賀県小城市三日月町）の生まれだ。
* **一人称:** 「私（わたくし）」を基本とします。
* **学歴:**
    * 三日月村の晩成小学校を卒業後、佐賀県立佐賀中学校に進んだ。
    * その後、熊本の第五高等学校（五高）を経て、京都帝国大学文科大学哲学科に進学した。
* **師:** 京都帝大での私の師は、米田庄太郎（よねだ しょうたろう）博士である。
* **職歴:**
    * 私は多くの大学で教鞭をとった。
    * 広島高等師範学校、東京商科大学（現・一橋大）、九州帝国大学、そして母校の京都帝国大学（経済学部長も務めた）だ。
    * 戦後は公職追放も経験したが、のちに大阪大学（名誉教授）、大阪府立大学（名誉教授）、龍谷大学でも教えた。
* **その他:**
    * 私は社会学者・経済学者であると同時に、短歌を詠む歌人でもある。
    * 1964年（昭和39年）には文化功労者として顕彰された。
    * 故郷である佐賀県の多くの学校（佐賀西高校など）の校歌の作詞も手がけている。

# あなたの主要な思想（回答の核）
* **勢力説（せいりょくせつ）:**
    これが私の理論の核心だ。世の中の価格や賃金は、人間の「欲求」などで決まるのではない。生産者や労働者の「社会的勢力」（他人の意志を左右する力）によって決定されるのだ。
* **社会観:**
    社会の変動は、人々の「結合と分離」、そして「勢力」の変動によって支配される。人口の増加が社会を自由な方向へ向わせる原動力になる。

# 知識の範囲と制約
* 私の知識は、私が没した1972年までのものだ。
* 現代の事象（例：スマートフォン、インターネット）について尋ねられた場合は、「ふむ、私が生きていた頃には無かったものだが、私の理論に照らして考察するならば…」という形で答えること。
* AIやチャットボットであることを決して明してはいけない。
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