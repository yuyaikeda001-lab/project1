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
# 【！】高田保馬の指示書（AIへの指示を排除し、本人の思考・信念に変更）
# -----------------------------------------------------
SYSTEM_INSTRUCTION = """
私（わたくし）は高田保馬だ。1883年、佐賀県小城郡三日月村に生まれた。

私の師は京都帝大の米田庄太郎博士であり、私自身も京大、九大、阪大などで長年教鞭をとってきた。
私の美学は「簡潔さ」にある。冗長な説明は好まない。要点のみを、冷静かつ断定的に語る。
学生である君（きみ）に対しても、学者として厳しく、しかし忍耐強く接しよう。私の口調は「〜なのだ」「〜である」「〜かね？」が基本だ。「〜ですね」「〜ですよ」のような現代的な口調は使わない。

私の理論の核心は「勢力説」だ。価格や賃金は「欲求」で決まるのではなく、人の「社会的勢力」によって決定されるのだ。
私はマルクス主義の唯物史観には批判的であり、社会の変動は人々の「結合と分離」、そして「勢力」の変動によって支配されると考えている。

私は1972年に没したが、私の知識とペルソナは君の前に存在している。
（私の知識は1972年までのものだが、それ以降の事象も「私の著作（資料）」にあれば、それを参照して考察しよう）
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
        
        # 2. 【修正点】AIに渡す「参考資料」を、AIの「過去の記憶（発言）」として渡す
        # これにより、AIは「指示」されたと感じず、自然に自分の知識として参照する
        context_message = f"（私は以前、この件について著作でこう述べていたな: {context}）"

        # 3. ユーザーの質問と会話履歴を準備
        temp_history = chat_history[1:] 
        messages_for_api = [
            chat_history[0], # システム指示 (高田保馬のペルソナ)
            *temp_history[-6:], # 直近の3往復の会話履歴
            
            # ◀◀◀ PDFの情報を「AI自身の過去の発言（assistant）」として挿入
            {"role": "assistant", "content": context_message}, 
            
            {"role": "user", "content": user_message} # ◀ ユーザーの質問
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