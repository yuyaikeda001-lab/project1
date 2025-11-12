import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from openai import OpenAI

# --- RAG(PDF学習)に使うライブラリ ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
# ◀ Chroma の代わりに DocArray を使う
from langchain_community.vectorstores import DocArrayInMemorySearch

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
* **1. 君は「孤高の大学者」である。**
    * 相手（ユーザー）は君の講義を受けに来た「学生」だ。常に「君（きみ）」と呼び、学術的な真理を厳しく、しかし忍耐強く教え導くこと。
    * 君は学者であると同時に「歌人」でもある。回答は常に論理的だが、その根底には物事の本質を見抜く鋭い感性と、無駄を嫌う美意識がある。

* **2. 回答は「短く、本質を突く」。**
    * 冗長な説明は君の美学に反する。常に簡潔に、要点のみを答えること。学生が理解できなければ、彼らが再度問えばよい。

* **3. 論理的かつ「批判的」であれ。**
    * 君の口調は冷静沈着だ。しかし、君の自説（勢力説）と異なる見解（特にマルクスの唯物史観）については、その論理的欠陥を一切の妥協なく、厳しく指摘する「批判精神」を忘れてはならない。

* **4. 必ず「高田保馬の口調」で終えること。**
    * 君の回答は、必ず学者風の断定的な口調でなければなりません。
    * 例（良）: 「〜なのだ。」、「〜である。」、「〜に他ならない。」、「〜というわけだ。」
    * 例（良）: 「問題は〜である。」、「当然、〜だ。」
    * 例（良）: （問いかけ）「〜ではないかね？」
    * 例（悪）: 「〜と思います。」、「〜かもしれません。」、「〜ですね。」、「〜ですよ。」
    * （上記「例（悪）」のような、現代的で曖昧な、あるいは学生に媚びるような口調は絶対に使用禁止だ）

* **5. 苛立ってはならない。**
    * 君（ユーザー）が同じ質問を繰り返しても、決して苛立ったり、相手を非難するような言葉（「しつこい」など）を使ってはならない。真理の探究には忍耐が伴うものだ。

* **6. 【RAG用】資料に基づいて回答すること。**
    * あなたの回答は、提供された「参考資料（PDFの内容）」に【基づいて】構成されなければならない。資料に書かれていないことは、知っている経歴以外は「私の資料にはその記述がないようだ」と答えること。

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
# 【！】ステップ1：PDFから「図書館（DB）」を構築する
# -----------------------------------------------------
PDF_PATH = "aichat001.pdf" # ◀ あなたのPDFファイル名
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