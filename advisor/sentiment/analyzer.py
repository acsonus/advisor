"""Yahoo Finance news sentiment analyzer using FinBERT."""

import os
from typing import Any
import pandas as pd
import yfinance as yf

try:
    from IPython.display import display
except ImportError:
    def display(x):
        """
        Goal:
            Provide a no-op fallback for IPython's `display()` function in standard CLI
            or headless execution environments where IPython is not installed.

        Execution Principle:
            Accepts any arbitrary object representation `x` and safely returns `None`
            without raising `NameError` or emitting unwanted output.
        """
        pass


class YahooSentiments:
    """
    Analyzer that fetches Yahoo Finance news and scores headlines using FinBERT.

    Manages news ingestion, NLP model caching, headline classification, and
    sentiment aggregation into trading signals and trend evaluations.
    """

    __api_key = None
    __cache: dict[str, Any] = {}
    __ticker_symbol: str = ""
    __yahoo_news: list[dict[str, Any]] = []

    _nlp_pipe = None

    @classmethod
    def _get_nlp_pipe(cls):
        """
        Lazily initialize and cache the HuggingFace FinBERT sequence classification pipeline.

        Goal:
        -----
        Provide shared, singleton access to the pretrained FinBERT financial sentiment model
        without incurring expensive model downloading and weight loading at module import time.

        Execution Principle:
        --------------------
        1. Check if `_nlp_pipe` class attribute is already instantiated. If present, return it immediately.
        2. If not instantiated:
           - Retrieve optional HuggingFace token from environment variable `HF_TOKEN`.
           - Load `BertForSequenceClassification` and `BertTokenizer` from `'yiyanghkust/finbert-tone'`.
           - Construct a HuggingFace `pipeline("text-classification")` with truncation enabled.
           - Cache the pipeline on `cls._nlp_pipe` and return.

        Returns:
        --------
        transformers.Pipeline
            Configured text classification pipeline for FinBERT tone scoring.
        """
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
        """
        Property accessor for the FinBERT sentiment analysis pipeline.

        Goal:
        -----
        Provide convenient instance-level access to the cached NLP pipeline.

        Execution Principle:
        --------------------
        Delegates directly to classmethod `_get_nlp_pipe()`.
        """
        return self._get_nlp_pipe()

    def __init__(self):
        """
        Initialize a YahooSentiments instance.

        Goal:
        -----
        Create empty state containers for parsed article entries and active ticker tracking.

        Execution Principle:
        --------------------
        Initializes `__news_data` to an empty list and `__ticker` to `None`.
        """
        self.__news_data: list[dict[str, Any]] = []
        self.__ticker = None

    def downloadYahooNews(self, ticker_symbol: str = 'AAPL') -> list[dict[str, Any]]:
        """
        Fetch recent news articles for a ticker symbol via Yahoo Finance.

        Goal:
        -----
        Query Yahoo Finance via `yfinance.Ticker` to retrieve raw news feeds containing
        titles, publisher info, and publication timestamps.

        Execution Principle:
        --------------------
        1. Store `ticker_symbol` in instance state `__ticker_symbol`.
        2. Create a `yf.Ticker(ticker_symbol)` object.
        3. Query `ticker.news`, falling back to an empty list if `None` is returned.
        4. Cache results in `__yahoo_news` and return the raw article list.

        Parameters:
        -----------
        ticker_symbol : str, default 'AAPL'
            Financial asset ticker symbol.

        Returns:
        --------
        list[dict[str, Any]]
            Raw list of article dictionaries returned by Yahoo Finance.
        """
        print("\n--- Fetching News from Yahoo Finance via yfinance ---")
        self.__ticker_symbol = ticker_symbol
        ticker = yf.Ticker(ticker_symbol)
        self.__yahoo_news = ticker.news or []
        return self.__yahoo_news

    def get_sentiment_score(self, headlines: list[str] | str) -> float:
        """
        Compute an aggregated sentiment score in the range [-1.0, 1.0] for a collection of headlines.

        Goal:
        -----
        Quantify the directional tone of one or more financial news titles into a normalized
        scalar value where positive represents optimism and negative represents pessimism.

        Execution Principle:
        --------------------
        1. Check if `headlines` is empty; return `0.0` if no headlines exist.
        2. Wrap single string input in a list.
        3. Pass headline strings to FinBERT pipeline `nlp_pipe`.
        4. For each classification result:
           - If label is 'Positive': add model probability score (`+res['score']`).
           - If label is 'Negative': subtract model probability score (`-res['score']`).
           - If label is 'Neutral': score contribution is zero.
        5. Return the cumulative score divided by the number of headlines.

        Parameters:
        -----------
        headlines : list[str] or str
            One or multiple news title strings.

        Returns:
        --------
        float
            Signed average sentiment score between -1.0 (strongly negative) and +1.0 (strongly positive).
        """
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
        """
        Parse fetched news articles, classify individual headlines, and compute signed scores.

        Goal:
        -----
        Transform raw nested Yahoo Finance news JSON objects into a structured tabular DataFrame
        containing publication dates, titles, FinBERT sentiment labels, and signed sentiment scores.

        Execution Principle:
        --------------------
        1. Reset internal news data cache `__news_data`.
        2. If `__yahoo_news` is populated:
           - Extract title and publication date (`pubDate`) for up to 300 articles.
           - Build initial DataFrame and extract title list.
           - Run each headline through the FinBERT pipeline, recording `'Sentiment_Label'` and `'Sentiment_Score'`.
           - Normalize `'Publication Date'` strings to timezone-naive UTC dates (`'Date'`) for time-series alignment.
           - Apply signed score mapping: positive scores keep sign, negative scores negated (`-score`), neutral set to 0.0.
        3. If no articles exist:
           - Return an empty DataFrame with expected columns `['Title', 'Publication Date', 'Sentiment_Label', 'Sentiment_Score', 'Date', 'Signed_Score']`.
        4. Compute and print overall sentiment across all extracted headlines.
        5. Return the processed DataFrame.

        Returns:
        --------
        pd.DataFrame
            Structured news sentiment DataFrame.
        """
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
                """
                Goal:
                    Compute a signed sentiment magnitude scalar in [-1.0, +1.0] from
                    FinBERT's discrete class label and positive probability confidence score.

                Execution Principle:
                    1. Check label category:
                       - If 'Positive', return `+score`.
                       - If 'Negative', return `-score`.
                       - For 'Neutral' or unrecognized labels, return `0.0`.
                """
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

    def check_bullish_trend(self, news_list: list[dict[str, Any]]) -> str:
        """
        Determine whether a collection of news articles confirms a strong bullish catalyst.

        Goal:
        -----
        Filter out superficial market chatter by requiring multiple distinctly positive
        articles and a significant average sentiment score.

        Execution Principle:
        --------------------
        1. If `news_list` is empty, return `'NEUTRAL / NOISY'`.
        2. Iterate over articles, scoring each article's title via `get_sentiment_score()`.
        3. Accumulate total score and increment `positive_count` for scores > 0.1.
        4. Evaluation criteria:
           - Requires at least 3 positive articles (`positive_count >= 3`).
           - AND an overall average score > 0.2 (`avg_score > 0.2`).
        5. If both conditions are satisfied, return `'BULLISH TREND CONFIRMED'`;
           otherwise return `'NEUTRAL / NOISY'`.

        Parameters:
        -----------
        news_list : list[dict[str, Any]]
            List of article dicts with `'title'` keys.

        Returns:
        --------
        str
            'BULLISH TREND CONFIRMED' or 'NEUTRAL / NOISY'.
        """
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

    def check_bearish_trend(self, news_list: list[dict[str, Any]]) -> str:
        """
        Determine whether news articles signal severe adverse risk or bearish trend confirmation.

        Goal:
        -----
        Identify potential corporate distress, fraud, or downgrades by applying asymmetric
        downward penalties to high-impact financial risk keywords.

        Execution Principle:
        --------------------
        1. If `news_list` is empty, return `'STABLE / RECOVERING'`.
        2. Define high-severity keywords: `['lawsuit', 'investigation', 'miss', 'downgrade', 'fraud', 'bankruptcy']`.
        3. For each article:
           - Score title tone using `get_sentiment_score()`.
           - If any red-flag keyword is present in lowercased title, apply penalty of `-0.2`.
           - If adjusted score < -0.1, increment `negative_count`.
        4. Evaluation criteria:
           - Requires at least 3 negative articles (`negative_count >= 3`).
           - AND average sentiment score < -0.2 (`avg_score < -0.2`).
        5. If both conditions are met, return `'BEARISH TREND CONFIRMED'`;
           otherwise return `'STABLE / RECOVERING'`.

        Parameters:
        -----------
        news_list : list[dict[str, Any]]
            List of article dicts with `'title'` keys.

        Returns:
        --------
        str
            'BEARISH TREND CONFIRMED' or 'STABLE / RECOVERING'.
        """
        negative_count = 0
        total_sentiment = 0.0
        red_flags = ['lawsuit', 'investigation', 'miss', 'downgrade', 'fraud', 'bankruptcy']

        for article in news_list:
            title = article['title'].lower()
            score = self.get_sentiment_score(title)

            if any(word in title for word in red_flags):
                score -= 0.2

            total_sentiment += score
            if score < -0.1:
                negative_count += 1

        if not news_list:
            return "STABLE / RECOVERING"

        avg_score = total_sentiment / len(news_list)
        if negative_count >= 3 and avg_score < -0.2:
            return "BEARISH TREND CONFIRMED"
        return "STABLE / RECOVERING"
