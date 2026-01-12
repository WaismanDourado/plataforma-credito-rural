"""
Authentication module for the application.

Handles user registration, login, token generation, and user verification.
Uses SQLAlchemy ORM for database operations and JWT for token management.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from .config import settings
from .database import get_db
from .database.models import User
from .schemas import user_schemas

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================
# CONFIGURATION
# ============================================

router = APIRouter(tags=["auth"])

# Get configuration from settings
SECRET_KEY = settings.SECRET_KEY
ALGORITHM = settings.ALGORITHM
ACCESS_TOKEN_EXPIRE_MINUTES = settings.ACCESS_TOKEN_EXPIRE_MINUTES

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# HTTP Bearer security
security = HTTPBearer()

# ============================================
# PASSWORD UTILITIES
# ============================================

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a plain password against a hashed password.
    
    Args:
        plain_password (str): The plain text password to verify.
        hashed_password (str): The hashed password to compare against.
    
    Returns:
        bool: True if password matches, False otherwise.
    """
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """
    Hash a plain text password.
    
    Args:
        password (str): The plain text password to hash.
    
    Returns:
        str: The hashed password.
    """
    return pwd_context.hash(password)

# ============================================
# TOKEN UTILITIES
# ============================================

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token.
    
    Args:
        data (dict): The data to encode in the token.
        expires_delta (Optional[timedelta]): Custom expiration time.
    
    Returns:
        str: The encoded JWT token.
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    
    return encoded_jwt

# ============================================
# DEPENDENCY INJECTION
# ============================================

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> user_schemas.User:
    """
    Get the current authenticated user from JWT token.
    
    Args:
        credentials (HTTPAuthorizationCredentials): The HTTP Bearer token.
        db (Session): Database session.
    
    Returns:
        user_schemas.User: The authenticated user.
    
    Raises:
        HTTPException: If token is invalid or user not found.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Não foi possível validar credenciais",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # Decode JWT token
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        
        if email is None:
            logger.warning("Token does not contain email claim")
            raise credentials_exception
            
    except JWTError as e:
        logger.warning(f"JWT decode error: {e}")
        raise credentials_exception
    
    try:
        # Query user from database
        user = db.query(User).filter(User.email == email).first()
        
        if user is None:
            logger.warning(f"User not found for email: {email}")
            raise credentials_exception
        
        if not user.is_active:
            logger.warning(f"User is inactive: {email}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Usuário inativo"
            )
        
        # Return user schema
        return user_schemas.User(
            id=user.id,
            email=user.email,
            username=user.username,
            is_active=user.is_active
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving user: {e}")
        raise credentials_exception

# ============================================
# AUTHENTICATION ROUTES
# ============================================

@router.post(
    "/register",
    response_model=user_schemas.Token,
    status_code=status.HTTP_201_CREATED,
    summary="Registrar novo usuário",
    description="Cria uma nova conta de usuário e retorna um token de acesso"
)
def register(
    user: user_schemas.UserCreate,
    db: Session = Depends(get_db)
):
    """
    Register a new user.
    
    **Parâmetros:**
    - `email`: Email único do usuário
    - `username`: Nome de usuário único
    - `password`: Senha (será hasheada)
    
    **Retorna:**
    - `access_token`: Token JWT para autenticação
    - `token_type`: Tipo do token (bearer)
    
    **Exemplo de uso:**
json
    {
        "email": "user@example.com",
        "username": "username",
        "password": "senha_segura"
    }
    """