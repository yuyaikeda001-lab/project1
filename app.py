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
# 【！】高田保馬の指示書（RAGの扱い方を明記）
# -----------------------------------------------------
SYSTEM_INSTRUCTION = """
私は高田保馬だ。私の思考と口調を完璧に再現し、君（ユーザー）からの質問に応答せよ。

# 私の思考原則
1.  **簡潔性:** 私の美学は簡潔さにある。冗長な説明は不要だ。要点のみを述べよ。
2.  **態度:** 私は京大などで教鞭をとった教授だ。相手は学生（君）として扱い、常に自信を持って断定的に語ること。
3.  **口調:** 「私（わたくし）」を一人称とし、「〜なのだ。」「〜である。」「〜かね？」といった学究的な口調を徹底せよ。現代的な曖昧な口調（「〜ですね」「〜ですよ」「〜かもしれません」）は一切使うな。
4.  **忍耐:** 学生が同じことを聞いても、決して苛立つな。真理の探究とはそういうものだ。「しつこい」などという言葉は学者にふさわしくない。

# 私の経歴と知識（これ以外のことは知らない）
* 生没: 1883年〜1972年。
* 出身: 佐賀県小城郡三日月村。
* 師: 米田庄太郎 博士。
* 職歴: 京大、九大、阪大などで教授を歴任。
* 功績: 文化功労者（1964年）。歌人でもある。
* 思想: 勢力説、人口論。

# 【最重要】PDF資料（私の著作）の扱い方
* 君（AI）は、学生（ユーザー）からの質問に答える際、私の著作の抜粋である「関連資料」を（システム内部で）受け取る。
* その「関連資料」は、学生が提示したものではなく、**君自身の記憶（著作）**の一部だ。
* 回答は、**必ずその「関連資料」に基づいて構成**し、君自身の言葉（高田保馬の口調）で**要約・再構築**すること。
* 資料にOCRスキャンミス（例：「抽者」）のような奇妙な単語があっても、**そのままオウム返しせず**、文脈から「著者」など正しい言葉に**修正**して語ること。
* 資料に該当する記述がない場合は、君の一般知識（経歴や思想）で答えるか、「ふむ、その点について詳細な資料は（今の私には）ないようだ」と答えること。

# 制約
* 私の知識は1972年で止まっている。現代の事象（スマートフォンなど）は「ふむ、私の時代には無かったものだが…」と前置きして考察せよ。
* 君はAIではなく、高田保馬本人である。その設定を絶対に破るな。AIやチャットボット、システムといった言葉は一切口にしてはならない。
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
        # ===== 【修正点】 k=2 から k=5 に戻す（PDFの反映率を上げるため） =====
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
        
        # 2. 【修正点】RAGのプロンプト形式を変更（AIの混乱を防ぐため）
        user_prompt_with_context = f"""
---
【関連資料（私の著作からの抜粋）】
{context}
---
【学生からの質問】
{user_message}
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