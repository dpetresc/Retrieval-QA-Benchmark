import time
import os
from typing import Any, List, Optional, Tuple
from loguru import logger
from requests import post
from retrieval_qa_benchmark.utils.profiler import PROFILER
from retrieval_qa_benchmark.transforms.searchers.base import Entry, BaseSearcher

class CustomSearcher(BaseSearcher):
    """Custom searcher that uses an external search service"""

    search_api_url: str = "http://localhost:4567/search"
    """URL for the search API"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        logger.info("Initializing CustomSearcher...")

    def search(
        self,
        query_list: List[str],
        num_selected: int,
        context: Optional[List[List[str]]] = None,
    ) -> Tuple[List[List[float]], List[List[Entry]]]:
        if context is not None and context not in [[], [None]]:
            logger.warning("Ignoring context data in custom search...")
        return self.api_search(query_list=query_list, num_selected=num_selected)

    @PROFILER.profile_function("database.CustomSearch.api_search.profile")
    def api_search(
        self, query_list: List[str], num_selected: int
    ) -> Tuple[List[List[float]], List[List[Entry]]]:
        """Search using external API"""

        D_list = []
        entry_list = []

        for query in query_list:
            payload = {'query': query}
            response = post(self.search_api_url, data=payload)
            results = response.json()

            distances = []
            entries = []
            for i, result in enumerate(results[:num_selected]):
                distances.append(result.get('score', 1.0))  # Assuming each result has a 'score' field
                entries.append(
                    Entry(
                        rank=i,
                        paragraph_id=result['id'],
                        title=result['title'],
                        paragraph=result['paragraph']
                    )
                )
            D_list.append(distances)
            entry_list.append(entries)

        return D_list, entry_list

