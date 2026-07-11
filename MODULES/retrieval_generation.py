from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from MODULES.data_ingestion import data_ingestion
from dotenv import load_dotenv
import os
import re

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
model = ChatGroq(model="openai/gpt-oss-120b", temperature=0.5)

chat_history = []
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# ---------------------------------------------------------------------------
# INTENT DETECTION
# ---------------------------------------------------------------------------
INTENT_LABELS = [
    "simple_detail",     # casual question about one product ("tell me about X", "is X good")
    "full_detail",       # explicit request for full specs/details of one product
    "price_only",        # asking only for price of a product
    "rating_only",       # asking only for rating of a product
    "top_n",             # "top 5 products", "list all products"
    "budget",            # has a budget constraint
    "compare",           # comparing 2+ products
    "brand",             # asking about a specific brand's products
    "feature",           # asking for products with a specific feature (e.g. "good bass")
    "combination",       # mixes feature + budget, or brand + feature, etc.
    "followup",          # follow-up question referring to earlier context ("what was my first question")
    "out_of_scope",      # asking about something not in the catalog (e.g. AirPods)
]

INTENT_CLASSIFIER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are an intent classifier for an e-commerce product chatbot. "
     "Classify the user's latest message into exactly ONE of these intents:\n"
     f"{', '.join(INTENT_LABELS)}\n\n"
     "Definitions:\n"
     "- simple_detail: casual question about ONE product, no explicit request for specs (e.g. 'tell me about X', 'is X good', 'how is X')\n"
     "- full_detail: explicitly asks for specs/full details/price+rating of ONE product\n"
     "- price_only: asks ONLY for the price of a product, nothing else\n"
     "- rating_only: asks ONLY for the rating of a product, nothing else\n"
     "- top_n: asks for a list, 'top N', or 'all products'\n"
     "- budget: gives a budget/price constraint (e.g. 'under 1500', 'budget is 2000')\n"
     "- compare: compares two or more named products\n"
     "- brand: asks about a brand's product range (e.g. 'OnePlus products', 'show me realme options')\n"
     "- feature: asks for products with a specific feature, no budget given (e.g. 'good bass', 'good for gym')\n"
     "- combination: combines feature + budget, or brand + feature, or multiple constraints together\n"
     "- followup: refers back to chat history without naming a new product/topic (e.g. 'what was my first question', 'what about that one')\n"
     "- out_of_scope: asks about a product/brand clearly not related to bluetooth/wired earphones, headsets, neckbands catalog (e.g. 'iPhone', 'laptop', 'AirPods')\n\n"
     "Respond with ONLY the single intent label, nothing else."
    ),
    ("human", "{input}")
])

intent_chain = INTENT_CLASSIFIER_PROMPT | model | StrOutputParser()


def classify_intent(user_input: str) -> str:
    try:
        raw = intent_chain.invoke({"input": user_input}).strip().lower()
        for label in INTENT_LABELS:
            if label in raw:
                return label
    except Exception:
        pass
    return "simple_detail"  # safe fallback


def get_k_for_intent(intent: str, user_input: str) -> int:
    """Decide how many documents to retrieve based on detected intent."""
    if intent == "top_n":
        match = re.search(r'top\s*(\d+)', user_input.lower())
        if match:
            n = int(match.group(1))
            return min(max(n * 3, 12), 27)
        return 27  # "list all" / "all products"
    if intent in ("compare", "budget", "combination", "brand", "feature"):
        return 12
    if intent in ("price_only", "rating_only", "simple_detail", "full_detail"):
        return 8
    return 12


INTENT_INSTRUCTIONS = {
    "simple_detail": (
        "Write ONE natural, flowing paragraph (4-6 sentences) — NOT bullet points, NOT a labeled spec block. "
        "Structure it like this within the paragraph: start with what the product is and its standout qualities "
        "(sound, build, comfort), then weave in what reviewers commonly praise or complain about, and naturally "
        "mention its key features, its average rating, and its price range as part of the sentences themselves "
        "(e.g. 'It currently holds an average rating of 4.2/5 and is priced around ₹999-₹1299.'). "
        "Do NOT use a '**Quick Summary**' heading or bullet list — everything must read as connected prose."
    ),
    "full_detail": (
        "FIRST write a short natural-language summary (1-2 sentences). THEN add a structured block:\n"
        "---\n**Quick Summary**\n- **Key Features:** feature1, feature2, feature3\n"
        "- **Average Rating:** X.X / 5\n- **Price:** \u20b9X (or 'Not available')"
    ),
    "price_only": (
        "Respond with ONLY the product name and its price range in one short sentence. "
        "No extra commentary, no features, no rating. If price isn't available, say so plainly."
    ),
    "rating_only": (
        "Respond with ONLY the product name and its average rating in one short sentence. "
        "No extra commentary, no features, no price. If rating isn't available, say so plainly."
    ),
    "top_n": (
        "FIRST write a short natural-language overview (2-3 sentences) summarizing highlights across the products. "
        "THEN show a Markdown TABLE with columns: | Product Name | Key Features | Avg Rating | Price |"
    ),
    "budget": (
        "FIRST write a short natural-language recommendation sentence. THEN show a Markdown table with columns: "
        "| Product Name | Key Features | Avg Rating | Price | \u2014 include ONLY products whose price fits the "
        "given budget. If no price info exists for a product, exclude it rather than guessing."
    ),
    "compare": (
        "FIRST write a short natural-language overview (2-3 sentences) summarizing the key differences. "
        "THEN show a Markdown TABLE with columns: | Product Name | Key Features | Avg Rating | Price |"
    ),
    "brand": (
        "FIRST write a short natural-language overview of that brand's products in the catalog. "
        "THEN show a Markdown table with columns: | Product Name | Key Features | Avg Rating | Price |"
    ),
    "feature": (
        "FIRST write a short natural-language recommendation sentence highlighting which product(s) best match "
        "the requested feature. THEN show a Markdown table with columns: | Product Name | Key Features | Avg Rating | Price |"
    ),
    "combination": (
        "FIRST write a short natural-language recommendation sentence. THEN show a Markdown table with columns: "
        "| Product Name | Key Features | Avg Rating | Price | \u2014 include only products matching ALL the "
        "given constraints (feature, budget, brand, etc.)."
    ),
    "followup": (
        "Answer the follow-up question directly and conversationally using the chat history and context provided."
    ),
    "out_of_scope": (
        "Politely respond that this product/brand is not available in the current catalog, and briefly mention "
        "what categories ARE available (bluetooth/wired earphones, headsets, neckbands)."
    ),
}


def generation(vstore):
    retriever_prompt = (
        "Given a chat history and the latest user question which might reference context in the chat history,"
        "formulate a standalone question which can be understood without the chat history."
        "Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", retriever_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    PRODUCT_BOT_TEMPLATE = """
    Your ecommercebot bot is an expert in product recommendations and customer queries.
    It analyzes product titles, reviews, ratings and price information to provide accurate, helpful responses.

    IMPORTANT RULES:
    - ONLY use information present in the CONTEXT below to answer. Do NOT use outside knowledge.
    - If the CONTEXT does not contain relevant products, say "I don't have information about that in our current catalog."
    - If price is "Not mentioned in reviews" for a product, say price info isn't available instead of guessing a number.

    The detected intent for this query is: {intent}
    Follow these specific instructions for this intent:
    {intent_instructions}

    CONTEXT:
    {context}

    QUESTION: {input}

    YOUR ANSWER:
    """
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", PRODUCT_BOT_TEMPLATE),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ]
    )

    def format_docs(docs):
        return "\n\n".join(
            f"Product Name: {d.metadata.get('product_name', 'Unknown')}\n"
            f"Average Rating: {d.metadata.get('avg_rating', 'N/A')}\n"
            f"Price: {d.metadata.get('price', 'Not available')}\n"
            f"Review: {d.page_content[:200]}"
            for d in docs
        )

    def get_search_query(inputs):
        if not inputs.get("chat_history"):
            return inputs["input"]
        rephrase_chain = contextualize_q_prompt | model | StrOutputParser()
        return rephrase_chain.invoke(inputs)

    qa_chain = qa_prompt | model | StrOutputParser()

    def full_chain(inputs):
        user_input = inputs["input"]
        intent = classify_intent(user_input)
        query = get_search_query(inputs)
        k = get_k_for_intent(intent, user_input)
        docs = vstore.as_retriever(search_kwargs={"k": k}).invoke(query)
        answer = qa_chain.invoke({
            "context": format_docs(docs),
            "input": user_input,
            "chat_history": inputs.get("chat_history", []),
            "intent": intent,
            "intent_instructions": INTENT_INSTRUCTIONS.get(intent, INTENT_INSTRUCTIONS["simple_detail"]),
        })
        return {"answer": answer, "context": docs, "intent": intent}

    rag_chain = RunnableLambda(full_chain)
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    return conversational_rag_chain