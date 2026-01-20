"""
V8 FINAL - LIVE WEBSOCKET
==========================
Real-time candle stream with auto-reconnect.
"""
import asyncio
import json
import websockets
from datetime import datetime
import yaml


def load_config():
    try:
        with open('config.yaml') as f:
            return yaml.safe_load(f)
    except:
        return {}


class LiveFeed:
    def __init__(self, on_candle):
        self.config = load_config()
        self.callback = on_candle
        self.running = False
    
    def _build_url(self) -> str:
        symbols = self.config.get('exchange', {}).get('symbols', ['BTC/USDT'])
        timeframes = self.config.get('exchange', {}).get('timeframes', ['15m'])
        
        streams = []
        for s in symbols:
            sym = s.lower().replace('/', '')
            for tf in timeframes:
                streams.append(f"{sym}@kline_{tf}")
        
        base = self.config.get('websocket', {}).get('url', 'wss://stream.binance.com:9443/stream?streams=')
        return base + '/'.join(streams)
    
    async def connect(self):
        url = self._build_url()
        reconnect = self.config.get('websocket', {}).get('reconnect_delay', 3)
        
        self.running = True
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 📡 Connecting to WebSocket...")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 📊 Streams: {len(self.config.get('exchange', {}).get('symbols', []))} symbols × {len(self.config.get('exchange', {}).get('timeframes', []))} TFs")
        
        # SSL context to handle certificate issues
        import ssl
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        while self.running:
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=20, ssl=ssl_context) as ws:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Connected!")
                    
                    while self.running:
                        raw = await ws.recv()
                        msg = json.loads(raw)
                        
                        payload = msg.get('data', {})
                        k = payload.get('k', {})
                        
                        if k.get('x', False):
                            await self.callback(payload)
            
            except Exception as e:
                if self.running:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️ Error: {e}, reconnecting...")
                    await asyncio.sleep(reconnect)
    
    async def stop(self):
        self.running = False
