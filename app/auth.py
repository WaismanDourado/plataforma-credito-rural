from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
from typing import Optional
import sqlite3

router = APIRouter(prefix="/auth", tags=["auth"])

SECRET_KEY = "sua-chave-secreta-super-segura-aqui-mude-em-prod"  # Mude para env var
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class UserCreate(BaseModel):
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class User(BaseModel):
    email: str

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
    return User(email=email)

@router.post("/register", response_model=Token)
def register(user: UserCreate, conn = Depends(get_db)):
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS users (email TEXT PRIMARY KEY, hashed_password TEXT)")
    cursor.execute("SELECT email FROM users WHERE email = ?", (user.email,))
    if cursor.fetchone():
        raise HTTPException(status_code=400, detail="Email já cadastrado")
    hashed_password = get_password_hash(user.password)
    cursor.execute("INSERT INTO users (email, hashed_password) VALUES (?, ?)", (user.email, hashed_password))
    conn.commit()
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={"sub": user.email}, expires_delta=access_token_expires)
    return Token(access_token=access_token, token_type="bearer")

@router.post("/login", response_model=Token)
def login(form_data: UserLogin, conn = Depends(get_db)):
    cursor = conn.cursor()
    cursor.execute("SELECT hashed_password FROM users WHERE email = ?", (form_data.email,))
    db_user = cursor.fetchone()
    if not db_user or not verify_password(form_data.password, db_user[0]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Credenciais incorretas")
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={"sub": form_data.email}, expires_delta=access_token_expires)
    return Token(access_token=access_token, token_type="bearer")

@router.get("/me", response_model=User)
def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user