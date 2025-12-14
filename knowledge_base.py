#!/usr/bin/env python3
"""
Knowledge Base - Interface to external knowledge sources
Integrates with Wikipedia, scientific databases, web search, and knowledge graphs
"""

import os
import json
import time
from typing import Dict, List, Optional, Any
import numpy as np

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    requests = None

try:
    import wikipedia
    WIKIPEDIA_AVAILABLE = True
except ImportError:
    WIKIPEDIA_AVAILABLE = False
    wikipedia = None

from language_processor import LanguageProcessor


class KnowledgeBase:
    """Interface to external knowledge sources"""
    
    def __init__(self, cache_dir: str = "knowledge_cache"):
        """
        Initialize knowledge base
        
        Args:
            cache_dir: Directory to cache knowledge queries
        """
        self.cache_dir = cache_dir
        self.cache: Dict[str, Dict] = {}
        
        # Initialize language processor
        self.lang_processor = LanguageProcessor()
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load existing cache
        self._load_cache()
        
        # Initialize Wikipedia if available
        if WIKIPEDIA_AVAILABLE:
            try:
                wikipedia.set_lang("en")
                print("✅ Wikipedia integration available")
            except Exception as e:
                print(f"⚠️  Wikipedia initialization error: {e}")
    
    def _load_cache(self):
        """Load cached knowledge from disk"""
        cache_file = os.path.join(self.cache_dir, "knowledge_cache.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    self.cache = json.load(f)
            except Exception as e:
                print(f"⚠️  Error loading cache: {e}")
                self.cache = {}
    
    def _save_cache(self):
        """Save cache to disk"""
        cache_file = os.path.join(self.cache_dir, "knowledge_cache.json")
        try:
            with open(cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            print(f"⚠️  Error saving cache: {e}")
    
    def _cache_key(self, query: str, source: str) -> str:
        """Generate cache key"""
        import hashlib
        key = f"{source}:{query.lower().strip()}"
        return hashlib.md5(key.encode()).hexdigest()
    
    def query_wikipedia(self, query: str, sentences: int = 3) -> Dict[str, Any]:
        """
        Query Wikipedia for information
        
        Args:
            query: Search query
            sentences: Number of sentences to return
            
        Returns:
            Dictionary with Wikipedia information
        """
        cache_key = self._cache_key(query, "wikipedia")
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        if not WIKIPEDIA_AVAILABLE:
            return {
                'source': 'wikipedia',
                'query': query,
                'content': '[Wikipedia not available]',
                'error': 'wikipedia library not installed'
            }
        
        try:
            # Search for page
            search_results = wikipedia.search(query, results=1)
            if not search_results:
                result = {
                    'source': 'wikipedia',
                    'query': query,
                    'content': f'No Wikipedia page found for: {query}',
                    'found': False
                }
                self.cache[cache_key] = result
                return result
            
            # Get page content
            page = wikipedia.page(search_results[0])
            summary = wikipedia.summary(search_results[0], sentences=sentences)
            
            result = {
                'source': 'wikipedia',
                'query': query,
                'title': page.title,
                'content': summary,
                'url': page.url,
                'found': True
            }
            
            self.cache[cache_key] = result
            self._save_cache()
            return result
            
        except wikipedia.exceptions.DisambiguationError as e:
            # Handle disambiguation
            options = e.options[:5]  # Top 5 options
            result = {
                'source': 'wikipedia',
                'query': query,
                'content': f'Disambiguation: {", ".join(options)}',
                'options': options,
                'found': False
            }
            return result
            
        except Exception as e:
            result = {
                'source': 'wikipedia',
                'query': query,
                'content': f'Error: {str(e)}',
                'found': False,
                'error': str(e)
            }
            return result
    
    def query_arxiv(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """
        Query arXiv for scientific papers
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            Dictionary with arXiv results
        """
        cache_key = self._cache_key(query, "arxiv")
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        if not REQUESTS_AVAILABLE:
            return {
                'source': 'arxiv',
                'query': query,
                'papers': [],
                'error': 'requests library not available'
            }
        
        try:
            # arXiv API endpoint
            url = "http://export.arxiv.org/api/query"
            params = {
                'search_query': query,
                'start': 0,
                'max_results': max_results
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            # Parse XML response (simplified)
            import xml.etree.ElementTree as ET
            root = ET.fromstring(response.content)
            
            papers = []
            for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                title = entry.find('{http://www.w3.org/2005/Atom}title').text
                summary = entry.find('{http://www.w3.org/2005/Atom}summary').text
                papers.append({
                    'title': title,
                    'summary': summary[:500]  # Limit summary length
                })
            
            result = {
                'source': 'arxiv',
                'query': query,
                'papers': papers,
                'found': len(papers) > 0
            }
            
            self.cache[cache_key] = result
            self._save_cache()
            return result
            
        except Exception as e:
            result = {
                'source': 'arxiv',
                'query': query,
                'papers': [],
                'found': False,
                'error': str(e)
            }
            return result
    
    def query_conceptnet(self, concept: str, limit: int = 5) -> Dict[str, Any]:
        """
        Query ConceptNet knowledge graph
        
        Args:
            concept: Concept to query
            limit: Maximum number of relations
            
        Returns:
            Dictionary with ConceptNet relations
        """
        cache_key = self._cache_key(concept, "conceptnet")
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        if not REQUESTS_AVAILABLE:
            return {
                'source': 'conceptnet',
                'concept': concept,
                'relations': [],
                'error': 'requests library not available'
            }
        
        try:
            # ConceptNet API
            url = f"http://api.conceptnet.io/c/en/{concept}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            relations = []
            
            for edge in data.get('edges', [])[:limit]:
                start = edge.get('start', {}).get('label', '')
                end = edge.get('end', {}).get('label', '')
                rel = edge.get('rel', {}).get('label', '')
                weight = edge.get('weight', 0.0)
                
                relations.append({
                    'relation': rel,
                    'start': start,
                    'end': end,
                    'weight': weight
                })
            
            result = {
                'source': 'conceptnet',
                'concept': concept,
                'relations': relations,
                'found': len(relations) > 0
            }
            
            self.cache[cache_key] = result
            self._save_cache()
            return result
            
        except Exception as e:
            result = {
                'source': 'conceptnet',
                'concept': concept,
                'relations': [],
                'found': False,
                'error': str(e)
            }
            return result
    
    def search_knowledge(self, query: str, sources: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Search across multiple knowledge sources
        
        Args:
            query: Search query
            sources: List of sources to search (None = all available)
            
        Returns:
            Dictionary with results from all sources
        """
        if sources is None:
            sources = ['wikipedia', 'arxiv', 'conceptnet']
        
        results = {
            'query': query,
            'sources': {},
            'combined_content': []
        }
        
        # Extract key concepts from query
        keywords = self.lang_processor.extract_keywords(query, top_k=3)
        main_concept = keywords[0][0] if keywords else query.split()[0]
        
        # Query each source
        if 'wikipedia' in sources:
            wiki_result = self.query_wikipedia(main_concept)
            results['sources']['wikipedia'] = wiki_result
            if wiki_result.get('found'):
                results['combined_content'].append(wiki_result.get('content', ''))
        
        if 'arxiv' in sources:
            arxiv_result = self.query_arxiv(query)
            results['sources']['arxiv'] = arxiv_result
            if arxiv_result.get('found'):
                for paper in arxiv_result.get('papers', []):
                    results['combined_content'].append(paper.get('summary', ''))
        
        if 'conceptnet' in sources:
            conceptnet_result = self.query_conceptnet(main_concept)
            results['sources']['conceptnet'] = conceptnet_result
            if conceptnet_result.get('found'):
                relations_text = ", ".join([
                    f"{r['start']} {r['relation']} {r['end']}"
                    for r in conceptnet_result.get('relations', [])
                ])
                results['combined_content'].append(relations_text)
        
        # Combine all content
        results['combined_text'] = "\n\n".join(results['combined_content'])
        
        # Convert to pattern for brain processing
        if results['combined_text']:
            results['pattern'] = self.lang_processor.text_to_pattern(results['combined_text']).tolist()
        else:
            results['pattern'] = []
        
        return results
    
    def get_knowledge_pattern(self, query: str) -> np.ndarray:
        """
        Get knowledge as neural pattern
        
        Args:
            query: Knowledge query
            
        Returns:
            Neural pattern representing knowledge
        """
        results = self.search_knowledge(query)
        
        if results.get('pattern'):
            return np.array(results['pattern'], dtype=np.float32)
        else:
            # Fallback: use query itself
            return self.lang_processor.text_to_pattern(query)
    
    def store_knowledge(self, knowledge: str, topic: str):
        """
        Store knowledge in cache for later retrieval
        
        Args:
            knowledge: Knowledge content
            topic: Topic/category
        """
        cache_key = self._cache_key(topic, "stored")
        
        self.cache[cache_key] = {
            'source': 'stored',
            'topic': topic,
            'content': knowledge,
            'timestamp': time.time()
        }
        
        self._save_cache()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'cache_size': len(self.cache),
            'wikipedia_available': WIKIPEDIA_AVAILABLE,
            'requests_available': REQUESTS_AVAILABLE
        }

