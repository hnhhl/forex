"""
Advanced Caching System
Ultimate XAU Super System V4.0
"""

import redis
import json
import pickle
from typing import Any, Optional
from datetime import datetime, timedelta

class CacheManager:
    """Advanced caching system for performance"""
    
    def __init__(self):
        try:
            self.redis_client = redis.Redis(
                host='localhost',
                port=6379,
                decode_responses=True
            )
        except:
            self.redis_client = None
            
        self.memory_cache = {}
        
    def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set cache value with TTL"""
        try:
            # Try Redis first
            if self.redis_client:
                serialized = json.dumps(value) if isinstance(value, (dict, list)) else str(value)
                return self.redis_client.setex(key, ttl, serialized)
            else:
                # Fallback to memory cache
                expiry = datetime.now() + timedelta(seconds=ttl)
                self.memory_cache[key] = {
                    'value': value,
                    'expiry': expiry
                }
                return True
        except Exception as e:
            print(f"Cache set error: {e}")
            return False
            
    def get(self, key: str) -> Optional[Any]:
        """Get cache value"""
        try:
            # Try Redis first
            if self.redis_client:
                value = self.redis_client.get(key)
                if value:
                    try:
                        return json.loads(value)
                    except:
                        return value
            else:
                # Check memory cache
                if key in self.memory_cache:
                    cache_item = self.memory_cache[key]
                    if datetime.now() < cache_item['expiry']:
                        return cache_item['value']
                    else:
                        del self.memory_cache[key]
                        
            return None
        except Exception as e:
            print(f"Cache get error: {e}")
            return None
            
    def delete(self, key: str) -> bool:
        """Delete cache key"""
        try:
            if self.redis_client:
                return bool(self.redis_client.delete(key))
            else:
                if key in self.memory_cache:
                    del self.memory_cache[key]
                    return True
            return False
        except Exception as e:
            print(f"Cache delete error: {e}")
            return False
            
    def clear_all(self) -> bool:
        """Clear all cache"""
        try:
            if self.redis_client:
                return self.redis_client.flushall()
            else:
                self.memory_cache.clear()
                return True
        except Exception as e:
            print(f"Cache clear error: {e}")
            return False

# Global cache manager
cache_manager = CacheManager()
