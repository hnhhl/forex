"""
Security Policies & Guidelines
Ultimate XAU Super System V4.0
"""

# Password Policy
PASSWORD_MIN_LENGTH = 12
PASSWORD_REQUIRE_UPPERCASE = True
PASSWORD_REQUIRE_LOWERCASE = True
PASSWORD_REQUIRE_NUMBERS = True
PASSWORD_REQUIRE_SYMBOLS = True
PASSWORD_EXPIRY_DAYS = 90

# Session Management
SESSION_TIMEOUT_MINUTES = 30
MAX_CONCURRENT_SESSIONS = 3
SESSION_RENEWAL_THRESHOLD = 5  # minutes

# API Security
API_RATE_LIMIT_PER_MINUTE = 100
API_RATE_LIMIT_PER_HOUR = 1000
API_KEY_EXPIRY_DAYS = 365
REQUIRE_API_KEY_ROTATION = True

# Encryption Standards
ENCRYPTION_ALGORITHM = "AES-256-GCM"
KEY_DERIVATION_FUNCTION = "PBKDF2"
HASH_ALGORITHM = "SHA-256"

# Audit Requirements
LOG_ALL_AUTH_ATTEMPTS = True
LOG_ALL_API_CALLS = True
LOG_ALL_TRADES = True
AUDIT_LOG_RETENTION_DAYS = 365

# Network Security
ALLOWED_IP_RANGES = [
    "10.0.0.0/8",
    "172.16.0.0/12", 
    "192.168.0.0/16"
]
REQUIRE_TLS_1_3 = True
DISABLE_WEAK_CIPHERS = True

# Data Protection
ENCRYPT_DATA_AT_REST = True
ENCRYPT_DATA_IN_TRANSIT = True
ANONYMIZE_LOGS = True
DATA_RETENTION_DAYS = 2555  # 7 years

class SecurityPolicyEnforcer:
    """Enforce security policies"""
    
    def __init__(self):
        self.policies_loaded = True
        
    def validate_password_policy(self, password: str) -> Dict:
        """Validate password against policy"""
        issues = []
        
        if len(password) < PASSWORD_MIN_LENGTH:
            issues.append(f"Password must be at least {PASSWORD_MIN_LENGTH} characters")
            
        if PASSWORD_REQUIRE_UPPERCASE and not any(c.isupper() for c in password):
            issues.append("Password must contain uppercase letters")
            
        if PASSWORD_REQUIRE_LOWERCASE and not any(c.islower() for c in password):
            issues.append("Password must contain lowercase letters")
            
        if PASSWORD_REQUIRE_NUMBERS and not any(c.isdigit() for c in password):
            issues.append("Password must contain numbers")
            
        if PASSWORD_REQUIRE_SYMBOLS and not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            issues.append("Password must contain symbols")
            
        return {
            'valid': len(issues) == 0,
            'issues': issues
        }
        
    def check_session_policy(self, session_start: datetime) -> bool:
        """Check if session complies with policy"""
        session_duration = datetime.now() - session_start
        return session_duration.total_seconds() < (SESSION_TIMEOUT_MINUTES * 60)

# Global policy enforcer
security_policy_enforcer = SecurityPolicyEnforcer()
