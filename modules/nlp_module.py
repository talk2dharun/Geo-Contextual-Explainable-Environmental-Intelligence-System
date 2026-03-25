"""
GEEIS - NLP Module
Handles news analysis including keyword extraction and sentiment analysis.
"""

import re
from collections import Counter
from textblob import TextBlob


# Environmental keywords for relevance scoring
ENVIRONMENTAL_KEYWORDS = [
    'pollution', 'contamination', 'toxic', 'chemical', 'bacteria',
    'nitrate', 'arsenic', 'lead', 'mercury', 'pesticide',
    'sewage', 'industrial', 'waste', 'discharge', 'runoff',
    'quality', 'treatment', 'purification', 'safe', 'unsafe',
    'health', 'disease', 'risk', 'hazard', 'dangerous',
    'river', 'lake', 'groundwater', 'aquifer', 'watershed',
    'regulation', 'standard', 'guideline', 'compliance', 'violation',
    'microplastic', 'heavy metal', 'e.coli', 'coliform', 'turbidity',
    'ph', 'dissolved', 'oxygen', 'chlorine', 'fluoride'
]


def extract_keywords(text, top_n=10):
    """
    Extract top keywords from text using frequency analysis.
    Filters out common stop words and retains domain-relevant terms.
    """
    if not text or not isinstance(text, str):
        return []

    # Common stop words
    stop_words = set([
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
        'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
        'would', 'could', 'should', 'may', 'might', 'shall', 'can',
        'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
        'as', 'into', 'through', 'during', 'before', 'after', 'above',
        'below', 'between', 'but', 'and', 'or', 'nor', 'not', 'so',
        'yet', 'both', 'either', 'neither', 'each', 'every', 'all',
        'any', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
        'only', 'own', 'same', 'than', 'too', 'very', 'just', 'that',
        'this', 'these', 'those', 'it', 'its', 'he', 'she', 'they',
        'them', 'their', 'his', 'her', 'our', 'your', 'my', 'which',
        'who', 'whom', 'what', 'when', 'where', 'why', 'how', 'if',
        'then', 'else', 'also', 'about', 'up', 'out', 'off', 'over',
        'under', 'again', 'further', 'once', 'here', 'there', 'says',
        'said', 'one', 'two', 'new', 'like', 'well', 'back', 'many',
        'still', 'even', 'us', 'get', 'since', 'now', 'long', 'make',
        'much', 'find', 'old', 'see', 'way', 'time', 'day', 'year',
        'made', 'come', 'around', 'per', 'part', 'take', 'first',
        'last', 'people', 'country', 'world', 'according', 'used',
        'going', 'use', 'could', 'would', 'while', 'been', 'chars',
        'http', 'https', 'www', 'com', 'also', 'however', 'reuters',
        'associated', 'press'
    ])

    # Clean and tokenize
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()

    # Filter
    filtered_words = [
        w for w in words
        if w not in stop_words and len(w) > 2
    ]

    # Count and rank
    word_counts = Counter(filtered_words)
    top_keywords = word_counts.most_common(top_n)

    return [{'keyword': word, 'count': count} for word, count in top_keywords]


def analyze_sentiment(text):
    """
    Perform sentiment analysis on text using TextBlob.
    Returns: polarity (-1 to 1), subjectivity (0 to 1), and label
    """
    if not text or not isinstance(text, str):
        return {'polarity': 0, 'subjectivity': 0, 'label': 'Neutral'}

    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    if polarity > 0.1:
        label = 'Positive'
    elif polarity < -0.1:
        label = 'Negative'
    else:
        label = 'Neutral'

    return {
        'polarity': round(polarity, 3),
        'subjectivity': round(subjectivity, 3),
        'label': label
    }


def analyze_news_articles(articles):
    """
    Analyze a list of news articles.
    Returns: analyzed articles with keywords, sentiment, and summary.
    """
    analyzed = []

    for article in articles:
        if article.get('source') == 'Error':
            continue

        # Combine title and description for analysis
        text = f"{article.get('title', '')} {article.get('description', '')} {article.get('content', '')}"

        # Extract keywords
        keywords = extract_keywords(text, top_n=5)

        # Sentiment analysis
        sentiment = analyze_sentiment(text)

        # Environmental relevance score
        relevance = calculate_relevance(text)

        analyzed.append({
            'title': article.get('title', 'N/A'),
            'source': article.get('source', 'N/A'),
            'published_at': article.get('published_at', 'N/A'),
            'url': article.get('url', ''),
            'keywords': keywords,
            'sentiment': sentiment,
            'relevance_score': relevance,
            'search_keyword': article.get('keyword', '')
        })

    return analyzed


def calculate_relevance(text):
    """
    Calculate environmental relevance score (0-100) based on 
    presence of domain-specific keywords.
    """
    if not text:
        return 0

    text_lower = text.lower()
    matches = sum(1 for kw in ENVIRONMENTAL_KEYWORDS if kw in text_lower)
    score = min(100, int((matches / len(ENVIRONMENTAL_KEYWORDS)) * 100 * 5))

    return score


def generate_news_summary(analyzed_articles):
    """
    Generate a structured summary from analyzed articles.
    Returns: summary dict with key insights.
    """
    if not analyzed_articles:
        return {
            'total_articles': 0,
            'average_sentiment': 'Neutral',
            'top_keywords': [],
            'key_findings': 'No relevant articles found.',
            'sentiment_distribution': {'Positive': 0, 'Neutral': 0, 'Negative': 0}
        }

    # Sentiment distribution
    sentiment_dist = {'Positive': 0, 'Neutral': 0, 'Negative': 0}
    total_polarity = 0
    all_keywords = []

    for article in analyzed_articles:
        sentiment = article.get('sentiment', {})
        label = sentiment.get('label', 'Neutral')
        sentiment_dist[label] = sentiment_dist.get(label, 0) + 1
        total_polarity += sentiment.get('polarity', 0)

        for kw in article.get('keywords', []):
            all_keywords.append(kw['keyword'])

    avg_polarity = total_polarity / len(analyzed_articles) if analyzed_articles else 0

    if avg_polarity > 0.1:
        avg_sentiment = 'Positive'
    elif avg_polarity < -0.1:
        avg_sentiment = 'Negative'
    else:
        avg_sentiment = 'Neutral'

    # Top keywords across all articles
    keyword_counts = Counter(all_keywords)
    top_keywords = [
        {'keyword': kw, 'frequency': count}
        for kw, count in keyword_counts.most_common(10)
    ]

    # Generate findings
    findings = []
    if sentiment_dist['Negative'] > sentiment_dist['Positive']:
        findings.append(
            "Recent news coverage shows predominantly negative sentiment "
            "regarding water quality issues."
        )
    elif sentiment_dist['Positive'] > sentiment_dist['Negative']:
        findings.append(
            "Recent news coverage shows generally positive developments "
            "in water quality management."
        )
    else:
        findings.append(
            "Recent news coverage shows mixed sentiment regarding water "
            "quality conditions."
        )

    high_relevance = [
        a for a in analyzed_articles
        if a.get('relevance_score', 0) > 30
    ]
    if high_relevance:
        findings.append(
            f"{len(high_relevance)} out of {len(analyzed_articles)} articles "
            f"show high environmental relevance."
        )

    return {
        'total_articles': len(analyzed_articles),
        'average_sentiment': avg_sentiment,
        'average_polarity': round(avg_polarity, 3),
        'top_keywords': top_keywords,
        'key_findings': ' '.join(findings),
        'sentiment_distribution': sentiment_dist
    }
