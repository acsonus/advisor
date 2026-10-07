"""Yahoo Finance news sentiment analyzer using FinBERT."""

import os
from typing import Any
import pandas as pd
import yfinance as yf

try:
    from IPython.display import display
except ImportError:
    def display(x):
        pass


class YahooSentiments:
    """Analyzer that fetches Yahoo Finance news and scores headlines using FinBERT."""

    __api_key = None
    __cache: dict[str, Any] = {}
    __ticker_symbol: str = ""
    __yahoo_news: list[dict[str, Any]] = []

    _nlp_pipe = None

    @classmethod
    def _get_nlp_pipe(cls):
        if cls._nlp_pipe is None:
            from transformers import BertForSequenceClassification, BertTokenizer, pipeline

            hf_token = os.getenv('HF_TOKEN')
            finbert = BertForSequenceClassification.from_pretrained(
                'yiyanghkust/finbert-tone', num_labels=3, token=hf_token
            )
            tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone', token=hf_token)
            cls._nlp_pipe = pipeline(
                "text-classification",
                model=finbert,
                tokenizer=tokenizer,
                token=hf_token,
                truncation=True,
            )
        return cls._nlp_pipe

    @property
    def nlp_pipe(self):
        return self._get_nlp_pipe()

    def __init__(self):
        self.__news_data: list[dict[str, Any]] = []
        self.__ticker = None

    def downloadYahooNews(self, ticker_symbol: str = 'AAPL'):
        print("\n--- Fetching News from Yahoo Finance via yfinance ---")
        self.__ticker_symbol = ticker_symbol
        ticker = yf.Ticker(ticker_symbol)
        self.__yahoo_news = ticker.news or []
        return self.__yahoo_news

    def get_sentiment_score(self, headlines):
        """Converts a list of headlines into a single score (-1 to 1)."""
        if not headlines:
            return 0.0
        pipe = self.nlp_pipe
        if isinstance(headlines, str):
            headlines = [headlines]
        results = pipe(headlines)
        score = 0.0
        for res in results:
            if res['label'] == 'Positive':
                score += res['score']
            elif res['label'] == 'Negative':
                score -= res['score']
        return score / len(headlines)

    def analyze_news(self) -> pd.DataFrame:
        self.__news_data = []  # reset for fresh analysis
        if self.__yahoo_news:
            print(f"Found {len(self.__yahoo_news)} news articles for {self.__ticker_symbol}.")
            print("Collecting article titles and publication times for DataFrame...")
            for article in self.__yahoo_news[:300]:
                content = article.get('content', {}) if isinstance(article, dict) else {}
                title = content.get('title', 'No Title Available')
                pub_date_str = content.get('pubDate', 'No Date Available')
                self.__news_data.append({'Title': title, 'Publication Date': pub_date_str})

            df_news = pd.DataFrame(self.__news_data)
            print("\n--- Yahoo Finance News DataFrame ---")
            display(df_news)

            financial_headlines_yf = df_news['Title'].tolist()

            df_news['Sentiment_Label'] = None
            df_news['Sentiment_Score'] = None

            print("\n--- Performing Sentiment Analysis on DataFrame News ---")
            pipe = self.nlp_pipe
            for index, row in df_news.iterrows():
                headline = row['Title']
                if headline:
                    result = pipe(headline)[0]
                    df_news.loc[index, 'Sentiment_Label'] = result['label']
                    df_news.loc[index, 'Sentiment_Score'] = result['score']

            print("\n--- Yahoo Finance News DataFrame with Sentiment Results ---")
            display(df_news)

            df_news['Date'] = (
                pd.to_datetime(df_news['Publication Date'], utc=True)
                .dt.normalize()
                .dt.tz_localize(None)
            )

            def signed_score(row):
                if row['Sentiment_Label'] == 'Positive':
                    return row['Sentiment_Score']
                if row['Sentiment_Label'] == 'Negative':
                    return -row['Sentiment_Score']
                return 0.0

            df_news['Signed_Score'] = df_news.apply(signed_score, axis=1)
        else:
            print(f"No news found for {self.__ticker_symbol} from Yahoo Finance.")
            financial_headlines_yf = []
            df_news = pd.DataFrame(
                columns=['Title', 'Publication Date', 'Sentiment_Label', 'Sentiment_Score', 'Date', 'Signed_Score']
            )

        if financial_headlines_yf:
            print("\n--- Overall Sentiment Analysis of Fetched Yahoo Finance News ---")
            real_current_sentiment_yf = self.get_sentiment_score(financial_headlines_yf)
            print(f"Overall sentiment for Yahoo Finance headlines: {real_current_sentiment_yf:.2f}")

            if real_current_sentiment_yf > 0.5:
                print(f"BULLISH SIGNAL: Sentiment is {real_current_sentiment_yf:.2f}. Executing Buy.")
            elif real_current_sentiment_yf < -0.5:
                print(f"BEARISH SIGNAL: Sentiment is {real_current_sentiment_yf:.2f}. Executing Sell.")
            else:
                print(f"NEUTRAL: Sentiment is {real_current_sentiment_yf:.2f}. No trade criteria met based on sentiment thresholds.")
        else:
            print("No Yahoo Finance headlines to analyze sentiment for.")

        return df_news

    def check_bullish_trend(self, news_list):
        positive_count = 0
        total_sentiment = 0.0

        for article in news_list:
            score = self.get_sentiment_score(article['title'])
            total_sentiment += score
            if score > 0.1:
                positive_count += 1

        if not news_list:
            return "NEUTRAL / NOISY"

        avg_score = total_sentiment / len(news_list)
        if positive_count >= 3 and avg_score > 0.2:
            return "BULLISH TREND CONFIRMED"
        return "NEUTRAL / NOISY"

    def check_bearish_trend(self, news_list):
        negative_count = 0
        total_sentiment = 0.0
        red_flags = ['lawsuit', 'investigation', 'miss', 'downgrade', 'fraud', 'bankruptcy']

        for article in news_list:
            title = article['title'].lower()
            score = self.get_sentiment_score(title)
            total_sentiment += score

            if any(word in title for word in red_flags):
                score -= 0.2

            if score < -0.1:
                negative_count += 1

        if not news_list:
            return "STABLE / RECOVERING"

        avg_score = total_sentiment / len(news_list)
        if negative_count >= 3 and avg_score < -0.2:
            return "BEARISH TREND CONFIRMED"
        return "STABLE / RECOVERING"
