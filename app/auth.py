from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
from typing import Optional
import sqlite3

from .schemas import user_schemas

router = APIRouter(tags=["auth"])

SECRET_KEY = "sua-chave-secreta-super-segura-aqui-mude-em-prod"  # Mude para env var
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")



# SQLite setup (integre com seu test.db existente)
def get_db():
    conn = sqlite3.connect('test.db')
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

security = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Não foi possível validar credenciais",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    # ✅ Retornar um objeto User com os campos corretos
    return user_schemas.User(
        id=1,  # Você precisará buscar do banco de dados
        email=email,
        username="username",  # Você precisará buscar do banco de dados
        is_active=True
    )

@router.post("/register", response_model=user_schemas.Token)
def register(user: user_schemas.UserCreate, conn = Depends(get_db)):
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            username TEXT UNIQUE NOT NULL,
            hashed_password TEXT NOT NULL,
            is_active BOOLEAN DEFAULT 1
        )
    """)
    cursor.execute("SELECT email FROM users WHERE email = ?", (user.email,))
    if cursor.fetchone():
        raise HTTPException(status_code=400, detail="Email já cadastrado")
    hashed_password = get_password_hash(user.password)
    cursor.execute(
        "INSERT INTO users (email, username, hashed_password, is_active) VALUES (?, ?, ?, ?)", 
        (user.email, user.username, hashed_password, True)
    )
    conn.commit()
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={"sub": user.email}, expires_delta=access_token_expires)
    return user_schemas.Token(access_token=access_token, token_type="bearer")

@router.post("/login", response_model=user_schemas.Token)
def login(user: user_schemas.UserLogin, conn = Depends(get_db)):
    cursor = conn.cursor()
    cursor.execute("SELECT hashed_password FROM users WHERE email = ?", (user.email,))
    db_user = cursor.fetchone()
    
    if not db_user or not verify_password(user.password, db_user[0]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, 
            detail="Credenciais incorretas"
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, 
        expires_delta=access_token_expires
    )
    return user_schemas.Token(access_token=access_token, token_type="bearer")

@router.get("/me", response_model=user_schemas.User)
def read_users_me(current_user: user_schemas.User = Depends(get_current_user)):
    return current_user