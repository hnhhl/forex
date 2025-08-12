"""
Security Manager & Hardening
Ultimate XAU Super System V4.0
"""

import hashlib
import jwt
import secrets
from datetime import datetime, timedelta
from typing import Dict, Optional
import bcrypt

class SecurityManager:
    """Comprehensive security management"""
    
    def __init__(self):
        self.secret_key = "your-super-secret-key"
        self.jwt_secret = "your-jwt-secret"
        self.failed_attempts = {}
        
    def hash_password(self, password: str) -> str:
        """Hash password securely"""
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
        
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
        
    def generate_api_key(self) -> str:
        """Generate secure API key"""
        return secrets.token_urlsafe(32)
        
    def create_jwt_token(self, user_data: Dict, expires_hours: int = 24) -> str:
        """Create JWT token"""
        payload = {
            **user_data,
            'exp': datetime.utcnow() + timedelta(hours=expires_hours),
            'iat': datetime.utcnow()
        }
        return jwt.encode(payload, self.jwt_secret, algorithm='HS256')
        
    def verify_jwt_token(self, token: str) -> Optional[Dict]:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
            
    def check_rate_limit(self, identifier: str, max_attempts: int = 5, 
                        window_minutes: int = 15) -> bool:
        """Check rate limiting"""
        now = datetime.now()
        
        if identifier not in self.failed_attempts:
            self.failed_attempts[identifier] = []
            
        # Clean old attempts
        cutoff = now - timedelta(minutes=window_minutes)
        self.failed_attempts[identifier] = [
            attempt for attempt in self.failed_attempts[identifier]
            if attempt > cutoff
        ]
        
        return len(self.failed_attempts[identifier]) < max_attempts
        
    def record_failed_attempt(self, identifier: str):
        """Record failed attempt"""
        if identifier not in self.failed_attempts:
            self.failed_attempts[identifier] = []
            
        self.failed_attempts[identifier].append(datetime.now())
        
    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        # Simple encryption (use proper encryption in production)
        import base64
        encoded = base64.b64encode(data.encode('utf-8'))
        return encoded.decode('utf-8')
        
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        import base64
        decoded = base64.b64decode(encrypted_data.encode('utf-8'))
        return decoded.decode('utf-8')
        
    def validate_input(self, input_data: str, input_type: str = 'general') -> bool:
        """Validate and sanitize input"""
        if not input_data:
            return False
            
        # Remove potentially dangerous characters
        dangerous_chars = ['<', '>', '"', "'", '&', ';', '(', ')', '|', '`']
        
        for char in dangerous_chars:
            if char in input_data:
                return False
                
        # Type-specific validation
        if input_type == 'email':
            import re
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            return bool(re.match(email_pattern, input_data))
            
        elif input_type == 'numeric':
            try:
                float(input_data)
                return True
            except ValueError:
                return False
                
        return True

# Global security manager
security_manager = SecurityManager()
