"""
Real-time WebSocket Service
Ultimate XAU Super System V4.0
"""

import asyncio
import websockets
import json
import logging
from typing import Dict, List, Callable

logger = logging.getLogger(__name__)

class RealTimeService:
    """Real-time data service using WebSockets"""
    
    def __init__(self):
        self.connections = set()
        self.subscribers = {}
        self.price_feeds = {}
        self.running = False
        
    async def start_server(self, host: str = "localhost", port: int = 8765):
        """Start WebSocket server"""
        logger.info(f"Starting WebSocket server on {host}:{port}")
        
        async def handler(websocket, path):
            await self.handle_connection(websocket)
            
        self.running = True
        server = await websockets.serve(handler, host, port)
        
        # Start price feed simulation
        asyncio.create_task(self.price_feed_loop())
        
        return server
        
    async def handle_connection(self, websocket):
        """Handle new WebSocket connection"""
        self.connections.add(websocket)
        logger.info(f"New connection: {websocket.remote_address}")
        
        try:
            async for message in websocket:
                await self.handle_message(websocket, message)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.connections.remove(websocket)
            logger.info(f"Connection closed: {websocket.remote_address}")
            
    async def handle_message(self, websocket, message: str):
        """Handle incoming message"""
        try:
            data = json.loads(message)
            msg_type = data.get('type')
            
            if msg_type == 'subscribe':
                await self.handle_subscription(websocket, data)
            elif msg_type == 'unsubscribe':
                await self.handle_unsubscription(websocket, data)
            elif msg_type == 'ping':
                await self.send_pong(websocket)
                
        except json.JSONDecodeError:
            await self.send_error(websocket, "Invalid JSON")
        except Exception as e:
            logger.error(f"Message handling error: {e}")
            
    async def handle_subscription(self, websocket, data: Dict):
        """Handle subscription request"""
        channel = data.get('channel')
        if channel not in self.subscribers:
            self.subscribers[channel] = set()
            
        self.subscribers[channel].add(websocket)
        
        # Send confirmation
        await self.send_message(websocket, {
            'type': 'subscription_confirmed',
            'channel': channel
        })
        
    async def handle_unsubscription(self, websocket, data: Dict):
        """Handle unsubscription request"""
        channel = data.get('channel')
        if channel in self.subscribers:
            self.subscribers[channel].discard(websocket)
            
    async def send_pong(self, websocket):
        """Send pong response"""
        await self.send_message(websocket, {
            'type': 'pong',
            'timestamp': asyncio.get_event_loop().time()
        })
        
    async def send_message(self, websocket, message: Dict):
        """Send message to websocket"""
        try:
            await websocket.send(json.dumps(message))
        except websockets.exceptions.ConnectionClosed:
            pass
            
    async def send_error(self, websocket, error_msg: str):
        """Send error message"""
        await self.send_message(websocket, {
            'type': 'error',
            'message': error_msg
        })
        
    async def broadcast_to_channel(self, channel: str, message: Dict):
        """Broadcast message to channel subscribers"""
        if channel not in self.subscribers:
            return
            
        message['channel'] = channel
        disconnected = set()
        
        for websocket in self.subscribers[channel]:
            try:
                await self.send_message(websocket, message)
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(websocket)
                
        # Remove disconnected websockets
        for websocket in disconnected:
            self.subscribers[channel].discard(websocket)
            
    async def price_feed_loop(self):
        """Simulate real-time price feed"""
        import random
        base_price = 2000.0
        
        while self.running:
            # Simulate price movement
            change = random.uniform(-2, 2)
            base_price += change
            
            price_data = {
                'type': 'price_update',
                'symbol': 'XAUUSD',
                'price': round(base_price, 2),
                'change': round(change, 2),
                'timestamp': asyncio.get_event_loop().time()
            }
            
            await self.broadcast_to_channel('prices', price_data)
            await asyncio.sleep(1)  # Update every second

# Global real-time service
realtime_service = RealTimeService()
