"""Module for download tweets by user."""
from time import sleep
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import tweepy

from src.sources.base_source import BaseSource


class Twitter(BaseSource):
    """Twitter source for download text(tweets)."""

    tweet_fields = [
        "id",
        "text",
        "author_id",
        "context_annotations",
        "created_at",
        "entities",
        "lang",
        "public_metrics",
    ]
    sleep_delay = 0.3

    def connect(self) -> None:
        """Connect to twitter api."""
        self.log_step("'s creating connection")
        if self.creds.get("bearer_token") is None:
            raise KeyError("'bearer_token' not defined")
        self.client = tweepy.Client(self.creds.get("bearer_token"))

    def __get_userid_by_username__(self, username: str) -> Tuple[Optional[int], Dict[str, Any]]:
        """Get user id by user name via twitter api."""
        self.log_step("'s getting userid by username")

        user_response = self.client.get_user(username=username)
        user_data = user_response.data

        if user_data is None:
            user_errors: List[Dict[str, str]] = user_response.errors
            user_error = user_errors[0]
            return None, self.make_final_response(status=user_error.get("detail"))

        return user_data.id, self.make_final_response(result=user_data.id)

    def __download_step__(self, userid: int, page_token: Optional[str]) -> Tuple[Any, str]:
        """Download tweets from user page by page_token."""
        response = self.client.get_users_tweets(
            userid,
            max_results=100,
            pagination_token=page_token,
            tweet_fields=self.tweet_fields,
        )
        response_meta = response.meta
        response_data = response.data
        pagination_token = response_meta.get("next_token")
        return response_data, pagination_token

    def download(self, username: str) -> pd.DataFrame:
        """Download user tweets by user name."""
        userid, user_response = self.__get_userid_by_username__(username)
        self.log_step("'s starting downloading")
        if not userid:
            return user_response

        next_page_token = None
        tweets_with_meta: List[Dict[str, Any]] = []
        while True:

            response_data, next_page_token = self.__download_step__(userid, next_page_token)
            if next_page_token is None:
                break
            tweets_with_meta.extend(map(dict, response_data))

            sleep(self.sleep_delay)
        return pd.DataFrame(tweets_with_meta)
