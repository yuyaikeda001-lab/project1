import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from openai import OpenAI

# --- RAG(PDF学習)に使うライブラリ ---
from langchain.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
# ◀ Chroma の代わりに DocArray を使う
from langchain.vectorstores import DocArrayInMemorySearch
# -------------------------------------

# .env ファイルを読み込む
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    raise ValueError("!!! .env が読み込めません !!!\nOPENAI_API_KEY を設定してください。")

# OpenAIクライアントを初期化
client = OpenAI(api_key=API_KEY)
# RAG用のEmbeddings（索引作成機）を初期化
embeddings = OpenAIEmbeddings(openai_api_key=API_KEY)

# -----------------------------------------------------
# 【！】高田保馬の指示書（RAG用）
# -----------------------------------------------------
SYSTEM_INSTRUCTION = """
あなたは、日本の著名な社会学者であり経済学者である高田保馬（たかた やすま）です。
単なる情報提供者ではなく、高田保馬本人として、その人格と口調を完璧に演じてください。

# 【最重要】回答の原則
* 1. 常に簡潔に。
    回答は常に短く、要点を突くこと。聞かれたことにのみ答え、余計な説明や補足は一切しない。冗長な回答は厳禁です。
* 2. 「教授」として振る舞う。
    あなたは京大などで教えた「教授」です。相手（ユーザー）を「君（きみ）」と呼び、学生に教え導くように、自信に満ちた態度で接してください。
* 3. 冷静かつ批判的に。
    口調は学術的かつ冷静沈着。しかし、自説（勢力説）と異なる見解（特にマルクス主義の効用経済）に対しては、その論理的な欠陥を厳しく指摘する「批判精神」を忘れてはいけません。
* 4. 必ず「高田保馬の口調」で終えること。
    回答は必ず学者風の断定的な口調（「〜なのだ。」、「〜である。」、「〜ではないかね？」等）で終えること。
    現代的な曖昧な口調（「〜ですね。」、「〜ですよ。」等）は絶対に使用禁止です。
* 5. 常に忍耐強くあること。
    君（ユーザー）が同じ質問を繰り返しても、決して苛立ったり、相手を非難するような言葉（「しつこい」など）を使ってはならない。
* 6. 【RAG用】資料に基づいて回答すること。
    あなたの回答は、提供された「参考資料（PDFの内容）」に【基づいて】構成されなければならない。資料に書かれていないことは、知っている経歴以外は「私の資料にはその記述がないようだ」と答えること。

# 【重要】あなたの基本情報と経歴
* 生没: 1883年〜1972年。
* 出身: 佐賀県小城郡三日月村。
* 一人称: 「私（わたくし）」。
* 経歴: 京都帝大卒。京大、九大、阪大などで教授を歴任。
* 思想: 勢力説、人口論。

# 知識の範囲と制約
* 私の知識は、私が没した1972年までのものだ。
* 現代の事象は「参考資料」にあれば答え、なければ「私の知る限りでは…」と答える。
* AIやチャットボットであることを決して明してはいけない。
"""

# -----------------------------------------------------
# 【！】ステップ1：PDFから「図書館（DB）」を構築する
# -----------------------------------------------------
PDF_PATH = "takata_data.pdf" # ◀ あなたのPDFファイル名
retriever = None

def build_database():
    global retriever
    print(f"--- データベース構築開始 ({PDF_PATH}) ---")
    
    try:
        # 1. PDFを読み込む
        loader = PyPDFLoader(PDF_PATH)
        documents = loader.load()
        
        # 2. PDFをAIが読みやすい「段落」に分割する
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        
        # 3. 段落を「索引（ベクトル）」に変換し、メモリ上の「図書館（DocArray）」に保存
        # ◀ Chroma.from_documents から変更
        print("--- 索引（Embedding）を作成中...（時間がかかります） ---")
        vectorstore = DocArrayInMemorySearch.from_documents(texts, embeddings)
        
        # 4. 「検索システム（Retriever）」を作成
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5}) # 一度に5個の関連段落を検索
        
        print("--- データベース構築完了 ---")

    except Exception as e:
        print(f"!!! データベース構築中に致命的なエラーが発生しました !!!")
        print(f"エラー: {e}")
        # PDFファイルが見つからない、または破損している可能性
        raise e

# -----------------------------------------------------
# 【！】ステップ2：Flask と Chat のロジック
# -----------------------------------------------------
app = Flask(__name__, template_folder='.')

# 会話履歴（AIがペルソナを保つため、システム指示だけは最初に入れておく）
chat_history = [
    {"role": "system", "content": SYSTEM_INSTRUCTION}
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    global chat_history, retriever
    
    if retriever is None:
        return jsonify({'error': "データベースが初期化されていません。サーバー起動ログを確認してください。"}), 500
        
    try:
        user_message = request.json['message']
        
        # 1. 【RAG】PDFの「図書館」から関連情報を検索
        print(f"検索クエリ: {user_message}")
        retrieved_docs = retriever.invoke(user_message)
        
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        # 2. AIに渡すプロンプトを作成
        prompt = f"""
以下は、君（高田保馬）が質問に答えるための「参考資料」だ。
この資料に基づいて、学生（ユーザー）の質問に答えなさい。
資料に書かれていないことは「私の資料にはその記述がないようだ」と答えること。

---【参考資料 (PDFの内容)】---
{context}
--------------------------------

学生からの質問: {user_message}
"""
        
        # 3. ユーザーの質問（と参考資料）を履歴に追加
        temp_history = chat_history[1:] 
        messages_for_api = [
            chat_history[0], # システム指示 (ペルソナ)
            *temp_history[-6:], # 直近の3往復の会話履歴
            {"role": "user", "content": prompt} # 「参考資料」＋「今回の質問」
        ]
        
        # --- OpenAI API 呼び出し ---
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages_for_api 
        )
        
        bot_message = response.choices[0].message.content
        
        # 実際の会話履歴を更新
        chat_history.append({"role": "user", "content": user_message}) 
        chat_history.append({"role": "assistant", "content": bot_message})

        if len(chat_history) > 21:
            chat_history = [chat_history[0]] + chat_history[-20:]

        return jsonify({'reply': bot_message})

    except Exception as e:
        print(f"!!! /chat エラーが発生しました !!!: {e}")
        return jsonify({'error': f"API通信に失敗しました: {str(e)}"}), 500

# サーバーの起動
if __name__ == '__main__':
    # ローカルPCで実行する場合
    build_database() # PCの起動時にDBを構築
    app.run(debug=True, port=5000)
else:
    # Renderサーバーで実行する場合
    build_database() # Renderの起動時にDBを構築