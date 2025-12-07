
import time
import requests

class MusicGroupChecker:
    """Simple API checker for music groups"""
    
    def __init__(self, api_timeout: int = 3):
        self.api_url = "https://musicbrainz.org/ws/2/artist"
        self.api_timeout = api_timeout
        self.cache = {}
        
    def check_group(self, name: str) -> bool:
        """Check if a name is a music group using MusicBrainz API"""

        if name in self.cache:
            return self.cache[name]
        
        try:
            time.sleep(0.5)

            params = {
                'query': name,
                'fmt': 'json',
                'limit': 5
            }
            
            response = requests.get(
                self.api_url, 
                params=params, 
                timeout=self.api_timeout,
                headers={'User-Agent': 'MusicChecker/1.0'}
            )
            
            if response.status_code == 200:
                data = response.json()
                artists = data.get('artists', [])

                for artist in artists:
                    if artist.get('type') in ['Group', 'Band']:
                        artist_name = artist.get('name', '').lower()
                        query_name = name.lower()
                        if query_name in artist_name or artist_name in query_name:
                            self.cache[name] = True
                            return True
            
            self.cache[name] = False
            return False
            
        except:
            self.cache[name] = False
            return False
