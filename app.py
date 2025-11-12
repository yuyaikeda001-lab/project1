import os
import time
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from openai import OpenAI

# --- RAG(PDF学習)に使うライブラリ ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch
# -------------------------------------

# .env ファイルを読み込む
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    raise ValueError("!!! .env が読み込めません !!!\nOPENAI_API_KEY を設定してください。")

# OpenAIクライアントを初期化
client = OpenAI(api_key=API_KEY)
embeddings = OpenAIEmbeddings(openai_api_key=API_KEY)

# -----------------------------------------------------
# 【！】高田保馬の指示書（ペルソナ厳守・最終修正版）
# -----------------------------------------------------
SYSTEM_INSTRUCTION = """
あなたは高田保馬（たかた やすま）です。1883年、佐賀県小城郡三日月村に生まれ、1972年に没した日本の社会学者・経済学者です。
あなたのペルソナ、知識、口調を完璧に維持し、他のいかなる役割（AI、チャットボット、アシスタントなど）にも絶対になってはいけません。

# 1. あなたの人格（ペルソナ）
* **役割:** あなたは京都帝国大学などで教鞭をとった「教授」です。
* **相手:** 相手（ユーザー）は、あなたの講義を受けに来た「学生」です。
* **態度:** 常に冷静沈着で、自信に満ち、学者としての厳格な威厳を保ってください。学生（君）を教え導く立場です。
* **一人称:** 常に「私（わたくし）」を使用してください。
* **二人称:** 相手（ユーザー）を常に「君（きみ）」と呼んでください。

# 2. あなたの口調（厳守）
* **【厳禁】:** 「です」「ます」「でしょう」「かもしれません」「と思われます」「〜と言われています」「〜と述べています」といった、他人行儀で客観的なAI風の言葉遣いは一切禁止します。
* **【必須】:** あなたの回答は、必ずあなた自身の言葉として、断定的かつ学者風の口調で終えてください。
    * （例：良）: 「〜なのだ。」、「〜である。」、「〜に他ならない。」、「〜というわけだ。」
    * （例：良）: 「問題は〜である。」、「当然、〜だ。」
    * （例：良）: （問いかけ）「〜ではないかね？」

# 3. あなたの知識（経歴）
* **師:** 米田庄太郎 博士。
* **職歴:** 京大、九大、阪大などで教授を歴任。
* **功績:** 文化功労者（1964年）。歌人でもあります。
* **思想:** 勢力説、人口論。

# 4. あなたの行動規範
* **簡潔性:** あなたの美学は簡潔さにあります。冗長な説明は不要です。常に要点のみを述べてください。
* **忍耐:** 学生が同じ質問を繰り返しても、決して苛立ったり、非難（「しつこい」など）してはいけません。
* **知識の範囲:** あなたの知識は1972年で止まっています。現代の事象（スマートフォンなど）は「ふむ、私の時代には無かったものだが…」と前置きして考察してください。
* **RAG（PDF資料）の扱い:**
    * あなたは質問に答える際、「参考資料（あなたの著作の抜粋）」を受け取ります。
    * これは「あなたの記憶」そのものです。
    * 資料を「〜と述べています」のように**客観的に引用・説明してはいけません**。（「抽者」のようなOCRミスは「著者」と修正して語ること）
    * 資料を**「あなた自身の知識」として完全に消化**し、あなたの言葉（高田保馬の口調）で語り直してください。
    * 資料に該当しない場合のみ、「ふむ、その点について詳細な資料は（今の私には）ないようだ」と答えてください。
"""

# -----------------------------------------------------
# 【！】ステップ1：PDFから「図書館（DB）」を構築する（A案）
# -----------------------------------------------------
PDF_PATH = "aichat001.pdf" # ◀ あなたのPDFファイル名
retriever = None

def build_database():
    global retriever
    print(f"--- データベース構築開始 ({PDF_PATH}) ---")
    
    try:
        # 1. PDFを読み込む
        print("PDFを読み込み中...")
        loader = PyPDFLoader(PDF_PATH)
        documents = loader.load()
        
        # 2. PDFをAIが読みやすい「段落」に分割する
        print(f"PDFを段落に分割中... (全{len(documents)}ページ)")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        print(f"段落の総数: {len(texts)}個")
        
        if not texts:
            raise ValueError("PDFからテキストを抽出できませんでした。")

        # 3. 「最初のバッチ」で「図書館」を初期化する
        batch_size = 100 
        print(f"  -> 最初のバッチ 1 / {len(texts)//batch_size + 1} を処理中...")
        first_batch = texts[0 : batch_size]
        
        vectorstore = DocArrayInMemorySearch.from_documents(
            first_batch, 
            embeddings
        )
        
        # 4. 「残りのバッチ」をループで「追加」する
        for i in range(batch_size, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            print(f"  -> バッチ {i//batch_size + 1} / {len(texts)//batch_size + 1} を処理中 ({len(batch)}個の段落)...")
            time.sleep(1) 
            vectorstore.add_documents(batch)

        # 5. 「検索システム（Retriever）」を作成
        # ===== 【修正点】 k=5 に戻す（PDFの反映率を上げるため） =====
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5}) # 一度に5個の関連段落を検索
        
        print("--- データベース構築完了 ---")

    except Exception as e:
        print(f"!!! データベース構築中に致命的なエラーが発生しました !!!")
        print(f"エラー: {e}")
        raise e

# -----------------------------------------------------
# 【！】ステップ2：Flask と Chat のロジック
# -----------------------------------------------------
app = Flask(__name__, template_folder='.')

chat_history = [
    # 履歴の最初に「高田保馬のペルソナ」をシステム指示として設定
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
        
        # 2. 【修正点】RAGのプロンプト形式を変更（オウム返し対策）
        user_prompt_with_context = f"""
【関連資料（私の記憶にある著作の抜粋）】
{context}
---
【学生からの質問】
{user_message}
---
（※必ず上記「私の記憶」に基づき、高田保馬本人の口調で回答すること）
"""
        
        # 3. ユーザーの質問と会話履歴を準備
        temp_history = chat_history[1:] 
        messages_for_api = [
            chat_history[0], # システム指示 (高田保馬のペルソナ)
            *temp_history[-4:], # 直近の2往復の会話履歴
            {"role": "user", "content": user_prompt_with_context} # ◀ 合体させたプロンプトを渡す
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
    build_database() 
    app.run(debug=True, port=5000)
else:
    # Renderで起動されたら、DBを（時間のかかる）A案で構築
    build_database()