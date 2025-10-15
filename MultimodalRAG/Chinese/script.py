import os
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.graphs import Neo4jGraph

# --- 1. Neo4j 連線資訊 ---
NEO4J_URI = "neo4j+s://e6242b4d.databases.neo4j.io"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "VQHjs1qerQjRH-4dwiZL7fEVTzOYGFPI0WjSeb6_0hI"

# --- 2. Google Gemini API 設定 ---
# (建議使用環境變數來管理金鑰)
os.environ["GOOGLE_API_KEY"] = "AIzaSyBcwn_D28tGHLgfm8qOb1vZ_TX2AQ3civE"

# --- 3. 初始化 LLM 和 Graph Transformer ---
try:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0)
    llm_transformer = LLMGraphTransformer(llm=llm)
    print("Gemini 模型和 Graph Transformer 初始化成功。")
except Exception as e:
    print(f"初始化模型時發生錯誤: {e}")
    exit()

# --- 4. 準備要處理的文件 ---
# 您可以將任何想分析的文本放在這裡
document_text = """
儘管台積電地位領先，但在台灣及全球市場依然面對多方競爭。在晶圓代工領域，其主要的在地競爭者包括專注於成熟製程的聯華電子，以及力積電、世界先進、茂矽和漢磊等公司。而在後段製程，除了日月光投控外，力原材料與化學品供應鏈
晶圓製造的命脈來自穩定且高品質的原材料供應。中美晶是重要的矽晶圓供應商之一，而中砂則生產先進製程不可或缺的再生晶圓與高階鑽石碟。在化學品與特殊氣體方面，李長榮集團、長興材料、勝一、三福化工等提供製程所需的化學原料；台特化、晶呈科、兆捷等公司則專攻半導體特殊氣體。崇越石英提供關鍵的石英產品，而華立與崇越等材料通路商，則確保了如日本信越光阻液等高階材料的順暢供應。

封裝、測試與後段合作夥伴
在產業鏈的後端，封裝與測試是實現晶片功能的最後一哩路。全球封測龍頭日月光投控，與台積電在先進封裝領域既是緊密的合作夥伴，也存在一定的競爭關係。京元電子則專注於半導體測試，配合台積電的產能擴張。同時，台積電也透過轉投資公司如精材（影像感測器封裝）和采鈺（晶圓級光學膜），深化在後段製程的佈局。倉儲供應商迅得也為台積電的先進封裝廠區提供重要的後勤支援。

產業競爭格局
儘管台積電地位領先，但在台灣及全球市場依然面對多方競爭。在晶圓代工領域，其主要的在地競爭者包括專注於成熟製程的聯華電子，以及力積電、世界先進、茂矽和漢磊等公司。而在後段製程，除了日月光投控外，力成科技、南茂科技、欣銓科技、矽格等主要封測廠，也與台積電自有的先進封裝業務形成潛在的競爭態勢。

客戶關係與全球挑戰
台積電的成功與其龐大的客戶群密不可分。台灣頂尖的IC設計公司，如全球手機晶片龍頭聯發科，以及在網通和顯示驅動IC領域領先的瑞昱半導體、聯詠科技，均是台積電的長期重要客戶，形成了穩固的客戶關係。展望未來，台積電除了持續深化與供應鏈及客戶的合作，也必須應對來自三星（Samsung）、英特爾（Intel）以及日本Rapidus等全球級對手的激烈競爭，以鞏固其在全球半導體產業中的領導地位。成科技、南茂科技、欣銓科技、矽格等主要封測廠，也與台積電自有的先進封裝業務形成潛在的競爭態勢。
"""

documents = [Document(page_content=document_text)]
print(f"準備處理文件，共 {len(documents)} 份。")

# --- 5. 進行圖譜推斷與寫入 ---
try:
    # 連接到 Neo4j 資料庫
    graph = Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USER,
        password=NEO4J_PASSWORD
    )

    # 從文件推斷圖譜結構
    print("正在呼叫 LLM 進行圖譜推斷...")
    graph_documents = llm_transformer.convert_to_graph_documents(documents)
    print(f"成功推斷出 {len(graph_documents[0].nodes)} 個節點和 {len(graph_documents[0].relationships)} 條關係。")

    # 將推斷出的圖譜寫入資料庫
    print("正在將圖譜寫入 Neo4j 資料庫...")
    graph.add_graph_documents(graph_documents)
    print("------ 資料寫入成功！------")
    print("您現在可以啟動 Streamlit 應用來查看更新後的圖譜。")

except Exception as e:
    print(f"處理過程中發生錯誤: {e}")