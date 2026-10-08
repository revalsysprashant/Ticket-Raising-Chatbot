import os
import json
import uuid
import secrets
import hashlib
import binascii
import datetime
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, EmailStr
from sqlalchemy import create_engine, Column, String, DateTime, Text, Integer, Float
from sqlalchemy.orm import declarative_base, sessionmaker
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI
from dotenv import load_dotenv

# --- 1. SQLITE DATABASE SETUP WITH TABLES ---
DATABASE_URL = "sqlite:///./app.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class TicketModel(Base):
    __tablename__ = "tickets"
    ticket_id = Column(String, primary_key=True, index=True)
    email = Column(String, nullable=False)
    order_id = Column(String, default="N/A")
    issue = Column(Text, nullable=False)
    status = Column(String, default="Open")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class ConversationModel(Base):
    __tablename__ = "conversations"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, index=True, nullable=False)
    role = Column(String, nullable=False)
    message = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

class UserModel(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class SessionTokenModel(Base):
    __tablename__ = "session_tokens"
    token = Column(String, primary_key=True, index=True)
    email = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class ProductModel(Base):
    __tablename__ = "products"
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    price = Column(Float, nullable=False)
    keywords = Column(String, nullable=False)  # comma separated, used for matching chat text

Base.metadata.create_all(bind=engine)

# --- 2. AI CLIENT & CHROMADB SETUP ---
load_dotenv()

AI_MODEL_ENDPOINT = os.getenv("AI_MODEL_ENDPOINT", "https://llmapi.revalsys.com/v1")
AI_MODEL = os.getenv("AI_MODEL", "qwen3-coder:30b")
API_KEY = os.getenv("API_KEY", "RevalKey")

ai_client = OpenAI(
    base_url=AI_MODEL_ENDPOINT,
    api_key=API_KEY
)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_URL = os.getenv("QDRANT_URL", "")

qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

collection_name = "expanded_knowledge_base"

try:
    collection_info = qdrant_client.get_collection(collection_name)
    has_collection = True
except Exception:
    has_collection = False

if not has_collection:
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )
    
    # RICH EXPANDED KNOWLEDGE BASE
    kb_data = [
        "Return Policy: Products can be returned within 7 days of delivery for a full refund. Items must be unused and in original packaging.",
        "Refund Process: To initiate a refund, submit a support ticket with your order ID. Once approved, pack the item securely and ship it to our hub. Upon receiving and inspecting the item, your refund will be processed within 3 to 5 business days to your original payment method.",
        "Order Cancellation: Orders can be canceled within 2 hours of placement from your Account Dashboard under 'My Orders'. If shipped, it must follow the return process.",
        "Password Reset: Go to portal.company.com/reset or press Ctrl+Alt+Del on your corporate laptop to update credentials.",
        "Order Tracking: Check real-time order status in your Account Dashboard under the 'My Orders' tab using your Tracking Number.",
        "Payment Methods: We accept Credit/Debit Cards (Visa, Mastercard), UPI, Net Banking, and PayPal.",
        "Hardware Warranty Claim Process: All electronic hardware carries a 1-year manufacturer warranty. To claim warranty, raise a ticket with photos/videos of the defect. Our technical team will process a repair or replacement.",
        "Shipping Time & Charges: Standard delivery takes 3 to 5 business days. Free shipping applies to orders above $50. Express shipping takes 1 to 2 days.",
        "Exchange Policy: Items can be exchanged for size or color variations within 7 days of delivery, subject to stock availability."
    ]
    doc_vectors = embeddings.embed_documents(kb_data)
    points = [
        PointStruct(id=i, vector=vector, payload={"text": text})
        for i, (text, vector) in enumerate(zip(kb_data, doc_vectors))
    ]
    qdrant_client.upsert(collection_name=collection_name, points=points)

# --- 2b. PRODUCT CATALOG SEED (3 products) ---
PRODUCTS = [
    {
        "id": "P001",
        "name": "AudioMax Wireless Headphones",
        "description": "Over-ear active noise-cancelling headphones with 30-hour battery life and plush memory-foam ear cups.",
        "price": 79.99,
        "keywords": "headphone,headphones,audiomax,earphone",
    },
    {
        "id": "P002",
        "name": "PulseFit Smart Watch",
        "description": "Fitness smartwatch with heart-rate tracking, built-in GPS, and 7-day battery life.",
        "price": 129.99,
        "keywords": "watch,smartwatch,pulsefit,fitness watch",
    },
    {
        "id": "P003",
        "name": "EchoWave Bluetooth Speaker",
        "description": "Portable waterproof Bluetooth speaker with 12-hour playtime and deep bass.",
        "price": 49.99,
        "keywords": "speaker,echowave,bluetooth speaker",
    },
]

def seed_products(db_session):
    if db_session.query(ProductModel).count() == 0:
        for p in PRODUCTS:
            db_session.add(ProductModel(**p))
        db_session.commit()

_seed_db = SessionLocal()
try:
    seed_products(_seed_db)
finally:
    _seed_db.close()

# --- 3. SQLITE CONVERSATION DB HELPERS ---
def save_chat_message(db_session, session_id: str, role: str, message: str):
    record = ConversationModel(session_id=session_id, role=role, message=message)
    db_session.add(record)
    db_session.commit()

def load_recent_chat_history(db_session, session_id: str, limit: int = 6) -> List[Dict[str, str]]:
    records = db_session.query(ConversationModel)\
        .filter(ConversationModel.session_id == session_id)\
        .order_by(ConversationModel.timestamp.desc())\
        .limit(limit)\
        .all()

    history = []
    for r in reversed(records):
        history.append({"role": r.role, "content": r.message})
    return history

# --- 3b. PASSWORD HASHING (stdlib only, no extra dependency) ---
def hash_password(password: str) -> str:
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000)
    return binascii.hexlify(salt).decode() + ":" + binascii.hexlify(dk).decode()

def verify_password(password: str, stored: str) -> bool:
    try:
        salt_hex, dk_hex = stored.split(":")
        salt = binascii.unhexlify(salt_hex)
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000)
        return binascii.hexlify(dk).decode() == dk_hex
    except Exception:
        return False

def get_current_user(authorization: Optional[str] = Header(default=None)) -> str:
    """Reads 'Authorization: Bearer <token>' header and returns the logged-in user's email."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated. Please log in.")
    token = authorization.split(" ", 1)[1].strip()
    db = SessionLocal()
    try:
        session_row = db.query(SessionTokenModel).filter(SessionTokenModel.token == token).first()
        if not session_row:
            raise HTTPException(status_code=401, detail="Session expired or invalid. Please log in again.")
        return session_row.email
    finally:
        db.close()

# --- 3c. CANCELLATION FLOW HELPERS (rule-based, deterministic) ---
def wants_to_cancel_product(query: str) -> bool:
    q = query.lower()
    if "policy" in q:
        return False
    cancel_phrases = [
        "cancel my product", "cancel my order", "want to cancel",
        "i want to cancel", "cancel a product", "cancel an order",
        "cancel product", "cancel order", "how do i cancel",
    ]
    return any(phrase in q for phrase in cancel_phrases) or (
        "cancel" in q and ("product" in q or "order" in q or "my" in q)
    )

def match_product_from_text(query: str, db_session) -> Optional[ProductModel]:
    q = query.lower()
    products = db_session.query(ProductModel).all()
    for p in products:
        if p.name.lower() in q:
            return p
        for kw in p.keywords.split(","):
            kw = kw.strip()
            if kw and kw in q:
                return p
    return None

def cancellation_instructions(product: ProductModel) -> str:
    return (
        f"Sure — here's how to cancel your **{product.name}** order:\n\n"
        f"1. Go to your Account Dashboard and open **My Orders**.\n"
        f"2. Find the {product.name} order and select **Cancel Order**.\n"
        f"3. If it hasn't shipped yet, it will be cancelled instantly and refunded in full.\n"
        f"4. If it has already shipped, cancellation isn't available — instead, once it "
        f"arrives you can start a return within 7 days of delivery for a full refund.\n\n"
        f"Would you like me to raise a support ticket to help you cancel this order?"
    )

# --- 4. GROQ TOOL DEFINITION ---
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "trigger_support_ticket",
            "description": "Call this ONLY when the user explicitly requests to open/raise a ticket or confirms 'yes' to a prompt asking to raise a ticket.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "description": "Short issue summary derived from context."},
                    "order_id": {"type": "string", "description": "Order ID if provided."}
                },
                "required": ["reason"]
            }
        }
    }
]

# --- 5. FASTAPI APP & SCHEMAS ---
app = FastAPI(title="SQLite Persistent Chatbot")

class ChatRequest(BaseModel):
    session_id: str
    user_query: str

class TicketCreateRequest(BaseModel):
    user_email: EmailStr
    issue_description: str
    order_id: Optional[str] = "N/A"

class SignupRequest(BaseModel):
    email: EmailStr
    password: str

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    # Entry point is the login page.
    base_dir = os.path.dirname(os.path.abspath(__file__))
    html_path = os.path.join(base_dir, "static", "login.html")
    if not os.path.exists(html_path):
        raise HTTPException(status_code=404, detail="login.html not found.")
    return FileResponse(html_path, media_type="text/html")

@app.get("/chat")
def read_chat_page():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    html_path = os.path.join(base_dir, "static", "index.html")
    if not os.path.exists(html_path):
        raise HTTPException(status_code=404, detail="index.html not found.")
    return FileResponse(html_path, media_type="text/html")

# --- 5b. AUTH ENDPOINTS ---
@app.post("/api/signup")
def signup(request: SignupRequest):
    db = SessionLocal()
    try:
        existing = db.query(UserModel).filter(UserModel.email == request.email).first()
        if existing:
            raise HTTPException(status_code=400, detail="An account with this email already exists.")
        user = UserModel(email=request.email, password_hash=hash_password(request.password))
        db.add(user)
        db.commit()
        token = secrets.token_hex(32)
        db.add(SessionTokenModel(token=token, email=request.email))
        db.commit()
        return {"status": "success", "token": token, "email": request.email}
    finally:
        db.close()

@app.post("/api/login")
def login(request: LoginRequest):
    db = SessionLocal()
    try:
        user = db.query(UserModel).filter(UserModel.email == request.email).first()
        if not user or not verify_password(request.password, user.password_hash):
            raise HTTPException(status_code=401, detail="Invalid email or password.")
        token = secrets.token_hex(32)
        db.add(SessionTokenModel(token=token, email=request.email))
        db.commit()
        return {"status": "success", "token": token, "email": request.email}
    finally:
        db.close()

@app.post("/api/logout")
def logout(authorization: Optional[str] = Header(default=None)):
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ", 1)[1].strip()
        db = SessionLocal()
        try:
            row = db.query(SessionTokenModel).filter(SessionTokenModel.token == token).first()
            if row:
                db.delete(row)
                db.commit()
        finally:
            db.close()
    return {"status": "success"}

# --- 5c. PRODUCT CATALOG ENDPOINT ---
@app.get("/api/products")
def get_products():
    db = SessionLocal()
    try:
        products = db.query(ProductModel).all()
        return {
            "products": [
                {"id": p.id, "name": p.name, "description": p.description}
                for p in products
            ]
        }
    finally:
        db.close()

@app.post("/api/chat")
def chat_endpoint(request: ChatRequest, current_user: str = Depends(get_current_user)):
    db = SessionLocal()
    session_id = request.session_id
    user_query = request.user_query.strip()

    try:
        save_chat_message(db, session_id, "user", user_query)

        # --- CANCELLATION FLOW (handled deterministically, before hitting the LLM) ---
        matched_product = match_product_from_text(user_query, db)

        if matched_product and ("cancel" in user_query.lower()):
            reply_text = cancellation_instructions(matched_product)
            product_payload = {"id": matched_product.id, "name": matched_product.name}
            save_chat_message(db, session_id, "assistant", reply_text)

            def cancel_info_event(reply_text=reply_text, product_payload=product_payload):
                payload = {
                    "type": "cancel_info",
                    "reply": reply_text,
                    "product": product_payload,
                }
                yield f"data: {json.dumps(payload)}\n\n"

            return StreamingResponse(cancel_info_event(), media_type="text/event-stream")

        if wants_to_cancel_product(user_query) and not matched_product:
            products = db.query(ProductModel).all()
            reply_text = "Which product would you like to cancel? Please select one below:"
            products_payload = [
                {"id": p.id, "name": p.name, "description": p.description}
                for p in products
            ]
            save_chat_message(db, session_id, "assistant", reply_text)

            def product_select_event(reply_text=reply_text, products_payload=products_payload):
                payload = {
                    "type": "product_select",
                    "reply": reply_text,
                    "products": products_payload,
                }
                yield f"data: {json.dumps(payload)}\n\n"

            return StreamingResponse(product_select_event(), media_type="text/event-stream")

        # Increase limit to 3 for richer context retrieval
        query_vector = embeddings.embed_query(user_query)
        search_response = qdrant_client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=3
        )
        search_results = search_response.points
        retrieved_docs = [hit.payload.get("text", "") for hit in search_results] if search_results else []
        kb_context = "\n\n".join(retrieved_docs) if retrieved_docs else "No specific KB context found."

        recent_history = load_recent_chat_history(db, session_id, limit=6)

        # REFINED SYSTEM PROMPT (STRICT SEPARATION BETWEEN INFORMATIONAL QUESTIONS & TICKET CREATION)
        system_prompt = f"""You are an official customer support assistant.

Knowledge Base Context:
{kb_context}

CRITICAL RULES:
1. INFORMATIONAL QUESTIONS (e.g., "what is the refund process?", "how long does shipping take?", "what payment methods do you accept?", "tell me about warranty"):
   - ALWAYS answer directly and clearly using ONLY the Knowledge Base context.
   - DO NOT trigger the `trigger_support_ticket` tool for informational questions, even if a ticket was previously raised!

2. TICKET CREATION REQUESTS:
   - When a user explicitly asks to raise/open a ticket, or replies "yes"/"sure" to a question asking if they want a ticket raised, ONLY THEN call the `trigger_support_ticket` tool.
   - If the user is just asking how something works or asking for information, answer their question directly in text."""

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(recent_history)

        check_completion = ai_client.chat.completions.create(
            messages=messages,
            model=AI_MODEL,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.1
        )
        response_msg = check_completion.choices[0].message

        # IF TOOL IS TRIGGERED
        if response_msg.tool_calls:
            tool_call = response_msg.tool_calls[0]
            args = json.loads(tool_call.function.arguments)

            extracted_order = args.get("order_id", "")
            extracted_reason = args.get("reason") or user_query

            reply_text = "I've opened the support ticket request form below. Please fill in your Gmail, Order ID, and Problem Description to submit:"

            save_chat_message(db, session_id, "assistant", reply_text)

            def tool_event():
                payload = {
                    "type": "tool_call",
                    "reply": reply_text,
                    "order_id": extracted_order,
                    "reason": extracted_reason
                }
                yield f"data: {json.dumps(payload)}\n\n"

            return StreamingResponse(tool_event(), media_type="text/event-stream")

        # STANDARD TEXT ANSWER (Informational Response)
        def text_stream_generator():
            stream = ai_client.chat.completions.create(
                messages=messages,
                model=AI_MODEL,
                temperature=0.2,
                max_tokens=400,
                stream=True
            )
            full_reply = ""
            for chunk in stream:
                content = chunk.choices[0].delta.content or ""
                if content:
                    full_reply += content
                    payload = {"type": "token", "content": content}
                    yield f"data: {json.dumps(payload)}\n\n"

            save_chat_message(db, session_id, "assistant", full_reply)

        return StreamingResponse(text_stream_generator(), media_type="text/event-stream")

    finally:
        db.close()

# --- 6. SAVE TICKET TO SQLITE ---
@app.post("/api/raise-ticket")
def raise_ticket(ticket: TicketCreateRequest):
    db = SessionLocal()
    try:
        ticket_id = f"TICK-{uuid.uuid4().hex[:6].upper()}"
        new_ticket = TicketModel(
            ticket_id=ticket_id,
            email=ticket.user_email,
            order_id=ticket.order_id or "N/A",
            issue=ticket.issue_description,
            status="Open"
        )
        db.add(new_ticket)
        db.commit()
        db.refresh(new_ticket)

        return {
            "status": "success",
            "ticket_id": ticket_id,
            "message": f"Support ticket created successfully! Reference ID: {ticket_id}."
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to save ticket to database.")
    finally:
        db.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=7000, reload=True)