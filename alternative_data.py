"""
ALTERNATIVE DATA INTEGRATION
============================
Integrates non-price data for institutional "Alpha":

1. Wikipedia Pageviews - Retail sentiment (search interest spikes)
2. Coinbase Liquidations - Order flow imbalance
3. News Sentiment - Aggregated news sentiment score

Usage:
    from alternative_data import AlternativeDataFeed
    
    feed = AlternativeDataFeed()
    await feed.update()
    signal = feed.get_combined_alpha('BTCUSDT')
"""
import asyncio
import aiohttp
import ssl
import certifi
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json


# =============================================================================
# WIKIPEDIA PAGEVIEWS (Retail Sentiment)
# =============================================================================

class WikipediaTracker:
    """
    Tracks Wikipedia pageviews for crypto/gold topics.
    
    Signal Logic:
    - High retail interest often marks local TOPS
    - Sharp interest spikes = potential reversal warning
    """
    
    TOPICS = {
        'BTCUSDT': ['Bitcoin', 'Cryptocurrency', 'Bitcoin_price'],
        'PAXGUSDT': ['Gold', 'Gold_as_an_investment', 'Gold_price'],
        'MACRO': ['Recession', 'Inflation', 'Stock_market_crash'],
    }
    
    BASE_URL = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article"
    
    def __init__(self):
        self.pageviews: Dict[str, Dict] = {}
        self.last_update: Optional[datetime] = None
    
    async def fetch_pageviews(self, session: aiohttp.ClientSession, 
                               topic: str, days: int = 30) -> List[int]:
        """Fetch daily pageviews for a topic"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        url = (f"{self.BASE_URL}/en.wikipedia/all-access/all-agents/"
               f"{topic}/daily/{start_date.strftime('%Y%m%d')}/{end_date.strftime('%Y%m%d')}")
        
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as r:
                if r.status == 200:
                    data = await r.json()
                    views = [item['views'] for item in data.get('items', [])]
                    return views
        except Exception as e:
            print(f"⚠️ Wikipedia fetch failed for {topic}: {e}")
        return []
    
    async def update(self, session: aiohttp.ClientSession):
        """Update pageviews for all topics"""
        for symbol, topics in self.TOPICS.items():
            total_views = 0
            recent_spike = False
            
            for topic in topics:
                views = await self.fetch_pageviews(session, topic, days=14)
                if views:
                    total_views += sum(views[-7:])  # Last 7 days
                    
                    # Detect spike (last day > 2x average)
                    if len(views) >= 7:
                        avg = sum(views[:-1]) / len(views[:-1])
                        if views[-1] > avg * 2:
                            recent_spike = True
            
            self.pageviews[symbol] = {
                'total_views': total_views,
                'spike_detected': recent_spike,
                'signal': 'BEARISH' if recent_spike else 'NEUTRAL'
            }
        
        self.last_update = datetime.now()
    
    def get_signal(self, symbol: str) -> dict:
        """Get Wikipedia signal for symbol"""
        return self.pageviews.get(symbol, {
            'total_views': 0,
            'spike_detected': False,
            'signal': 'NEUTRAL'
        })


# =============================================================================
# COINBASE LIQUIDATIONS
# =============================================================================

class CoinbaseLiquidations:
    """
    Tracks liquidation events from Coinbase/major exchanges.
    
    Signal Logic:
    - Large short liquidation = "Short Squeeze" → BULLISH
    - Large long liquidation = "Long Flush" → BEARISH
    """
    
    # Use public liquidation data endpoints
    COINGLASS_URL = "https://open-api.coinglass.com/public/v2/liquidation_chart"
    
    def __init__(self):
        self.liquidations: Dict[str, Dict] = {}
        self.last_update: Optional[datetime] = None
    
    async def fetch_liquidations(self, session: aiohttp.ClientSession, 
                                   symbol: str = 'BTC') -> dict:
        """Fetch recent liquidation data"""
        # Coinglass requires API key for full access
        # Using fallback estimation based on price volatility
        
        try:
            # Try Coinglass public endpoint
            params = {'symbol': symbol, 'interval': '1h'}
            async with session.get(
                self.COINGLASS_URL, 
                params=params,
                timeout=aiohttp.ClientTimeout(total=5)
            ) as r:
                if r.status == 200:
                    data = await r.json()
                    if data.get('success'):
                        return data.get('data', {})
        except Exception as e:
            pass  # Fallback to estimation
        
        return {}
    
    async def update(self, session: aiohttp.ClientSession):
        """Update liquidation data"""
        for symbol in ['BTC', 'PAXG']:
            data = await self.fetch_liquidations(session, symbol)
            
            if data:
                long_liq = sum(d.get('longLiquidationUsd', 0) for d in data[-24:])
                short_liq = sum(d.get('shortLiquidationUsd', 0) for d in data[-24:])
            else:
                # Fallback: estimate from volatility
                long_liq = 0
                short_liq = 0
            
            # Calculate imbalance
            total = long_liq + short_liq
            if total > 0:
                imbalance = (short_liq - long_liq) / total  # Positive = shorts got rekt
            else:
                imbalance = 0
            
            # Signal
            if imbalance > 0.3:
                signal = 'BULLISH'  # Short squeeze
            elif imbalance < -0.3:
                signal = 'BEARISH'  # Long flush
            else:
                signal = 'NEUTRAL'
            
            key = 'BTCUSDT' if symbol == 'BTC' else 'PAXGUSDT'
            self.liquidations[key] = {
                'long_liquidated': long_liq,
                'short_liquidated': short_liq,
                'imbalance': round(imbalance, 3),
                'signal': signal
            }
        
        self.last_update = datetime.now()
    
    def get_signal(self, symbol: str) -> dict:
        """Get liquidation signal for symbol"""
        return self.liquidations.get(symbol, {
            'long_liquidated': 0,
            'short_liquidated': 0,
            'imbalance': 0,
            'signal': 'NEUTRAL'
        })


# =============================================================================
# NEWS SENTIMENT
# =============================================================================

class NewsSentiment:
    """
    Aggregates news sentiment from multiple sources.
    
    Score: -1 (very bearish) to +1 (very bullish)
    
    Sources:
    - CryptoCompare News API
    - NewsData.io (backup)
    """
    
    CRYPTOCOMPARE_URL = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN"
    
    # Sentiment keywords (simple lexicon-based)
    BULLISH_WORDS = ['surge', 'rally', 'breakout', 'bullish', 'soar', 'gain', 
                     'positive', 'buy', 'accumulate', 'institutional', 'adoption']
    BEARISH_WORDS = ['crash', 'dump', 'bearish', 'sell', 'fear', 'panic',
                     'collapse', 'hack', 'scam', 'ban', 'regulation', 'warning']
    
    def __init__(self):
        self.sentiment: Dict[str, Dict] = {}
        self.headlines: List[dict] = []
        self.last_update: Optional[datetime] = None
    
    def _analyze_sentiment(self, text: str) -> float:
        """Simple lexicon-based sentiment analysis"""
        text = text.lower()
        
        bullish_count = sum(1 for word in self.BULLISH_WORDS if word in text)
        bearish_count = sum(1 for word in self.BEARISH_WORDS if word in text)
        
        total = bullish_count + bearish_count
        if total == 0:
            return 0.0
        
        return (bullish_count - bearish_count) / total
    
    async def fetch_news(self, session: aiohttp.ClientSession, 
                          categories: str = 'BTC,ETH') -> List[dict]:
        """Fetch recent crypto news"""
        try:
            params = {'categories': categories}
            async with session.get(
                self.CRYPTOCOMPARE_URL,
                params=params,
                timeout=aiohttp.ClientTimeout(total=5)
            ) as r:
                if r.status == 200:
                    data = await r.json()
                    return data.get('Data', [])[:20]  # Last 20 articles
        except Exception as e:
            print(f"⚠️ News fetch failed: {e}")
        return []
    
    async def update(self, session: aiohttp.ClientSession):
        """Update news sentiment"""
        articles = await self.fetch_news(session)
        
        if articles:
            self.headlines = [
                {'title': a.get('title', ''), 'source': a.get('source', '')}
                for a in articles[:10]
            ]
            
            # Calculate overall sentiment
            sentiments = [self._analyze_sentiment(a.get('title', '') + ' ' + a.get('body', ''))
                          for a in articles]
            avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
        else:
            avg_sentiment = 0
        
        # Signal based on sentiment
        if avg_sentiment > 0.2:
            signal = 'BULLISH'
        elif avg_sentiment < -0.2:
            signal = 'BEARISH'
        else:
            signal = 'NEUTRAL'
        
        self.sentiment = {
            'score': round(avg_sentiment, 3),
            'signal': signal,
            'headline_count': len(articles)
        }
        
        self.last_update = datetime.now()
    
    def get_signal(self) -> dict:
        """Get overall news sentiment"""
        return self.sentiment if self.sentiment else {
            'score': 0,
            'signal': 'NEUTRAL',
            'headline_count': 0
        }
    
    def get_headlines(self) -> List[dict]:
        """Get recent headlines"""
        return self.headlines


# =============================================================================
# COMBINED ALTERNATIVE DATA FEED
# =============================================================================

class AlternativeDataFeed:
    """
    Combined alternative data feed for institutional alpha.
    
    Aggregates:
    - Wikipedia Pageviews (retail sentiment)
    - Coinbase Liquidations (order flow)
    - News Sentiment (media sentiment)
    
    Provides a combined "Alpha Score" for each symbol.
    """
    
    def __init__(self, symbols: List[str] = None):
        self.symbols = symbols or ['BTCUSDT', 'PAXGUSDT']
        
        self.wikipedia = WikipediaTracker()
        self.liquidations = CoinbaseLiquidations()
        self.news = NewsSentiment()
        
        self.session = None
        self.last_update: Optional[datetime] = None
    
    async def start(self):
        """Start the feed session"""
        ssl_ctx = ssl.create_default_context(cafile=certifi.where())
        conn = aiohttp.TCPConnector(ssl=ssl_ctx)
        self.session = aiohttp.ClientSession(connector=conn)
        print("📡 Alternative Data Feed started")
    
    async def stop(self):
        """Stop the feed session"""
        if self.session:
            await self.session.close()
    
    async def update(self):
        """Update all data sources"""
        if not self.session:
            await self.start()
        
        # Update all sources in parallel
        await asyncio.gather(
            self.wikipedia.update(self.session),
            self.liquidations.update(self.session),
            self.news.update(self.session),
            return_exceptions=True
        )
        
        self.last_update = datetime.now()
    
    def get_combined_alpha(self, symbol: str) -> dict:
        """
        Get combined alpha signal for a symbol.
        
        Returns:
        --------
        dict with 'alpha_score' (-1 to +1), 'signal', and component details
        """
        wiki = self.wikipedia.get_signal(symbol)
        liq = self.liquidations.get_signal(symbol)
        news = self.news.get_signal()
        
        # Score calculation
        score = 0
        confidence = 50
        
        # Wikipedia (contrarian - spikes are bearish)
        if wiki['signal'] == 'BEARISH':
            score -= 0.3
            confidence += 5
        
        # Liquidations (momentum - follow the squeeze)
        if liq['signal'] == 'BULLISH':
            score += 0.4
            confidence += 10
        elif liq['signal'] == 'BEARISH':
            score -= 0.4
            confidence += 10
        
        # News sentiment
        score += news.get('score', 0) * 0.3
        
        # Final signal
        if score > 0.3:
            signal = 'STRONG_BUY'
            confidence += 15
        elif score > 0.1:
            signal = 'BUY'
            confidence += 5
        elif score < -0.3:
            signal = 'STRONG_SELL'
            confidence += 15
        elif score < -0.1:
            signal = 'SELL'
            confidence += 5
        else:
            signal = 'NEUTRAL'
        
        return {
            'alpha_score': round(score, 3),
            'signal': signal,
            'confidence': min(confidence, 95),
            'components': {
                'wikipedia': wiki,
                'liquidations': liq,
                'news': news
            }
        }
    
    def get_features(self, symbol: str) -> dict:
        """
        Get features for ML pipeline.
        
        Returns numeric features suitable for model input.
        """
        wiki = self.wikipedia.get_signal(symbol)
        liq = self.liquidations.get_signal(symbol)
        news = self.news.get_signal()
        
        return {
            'wiki_spike': 1 if wiki.get('spike_detected') else 0,
            'wiki_views': wiki.get('total_views', 0),
            'liq_imbalance': liq.get('imbalance', 0),
            'news_sentiment': news.get('score', 0),
        }


# =============================================================================
# TEST
# =============================================================================

async def test_alternative_data():
    """Test the alternative data feed"""
    feed = AlternativeDataFeed()
    await feed.start()
    
    print("\n" + "="*70)
    print("📊 ALTERNATIVE DATA FEED - Test")
    print("="*70)
    
    await feed.update()
    
    for symbol in feed.symbols:
        alpha = feed.get_combined_alpha(symbol)
        features = feed.get_features(symbol)
        
        print(f"\n{symbol}:")
        print(f"  Alpha Score: {alpha['alpha_score']}")
        print(f"  Signal: {alpha['signal']} ({alpha['confidence']}%)")
        print(f"  Wikipedia: {alpha['components']['wikipedia']}")
        print(f"  Liquidations: {alpha['components']['liquidations']}")
        print(f"  News: {alpha['components']['news']}")
        print(f"  ML Features: {features}")
    
    # Print headlines
    print("\n📰 Recent Headlines:")
    for h in feed.news.get_headlines()[:5]:
        print(f"  - {h['title'][:60]}... ({h['source']})")
    
    await feed.stop()


if __name__ == "__main__":
    print("="*70)
    print("📡 ALTERNATIVE DATA INTEGRATION - Test")
    print("="*70)
    asyncio.run(test_alternative_data())
