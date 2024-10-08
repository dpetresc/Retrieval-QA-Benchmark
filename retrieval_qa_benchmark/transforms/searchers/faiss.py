import os
from typing import Any, List, Optional, Tuple

import faiss
import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

from retrieval_qa_benchmark.utils.profiler import PROFILER

from .base import Entry, PluginVectorSearcher


class FaissSearcher(PluginVectorSearcher):
    """FAISS searcher"""

    embedding_name: str
    """embedding model name"""
    index_path: str
    """path to faiss dumped index"""
    nprobe: int = 128
    """number of clusters to search for IVF indices"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        logger.info("load index...")
        self.index = faiss.read_index(self.index_path)
        logger.info("load mpnet model...")
        self.model = SentenceTransformer(self.embedding_name)

    def search(
        self,
        query_list: list,
        num_selected: int,
        context: Optional[List[List[str]]] = None,
    ) -> Tuple[List[List[float]], List[List[Entry]]]:
        if context is not None and context not in [[], [None]]:
            logger.warning("Ignoring context data in faiss search...")
        return self.emb_filter(query_list=query_list, num_selected=num_selected)

    @PROFILER.profile_function("database.FaissSearch.emb_filter.profile")
    def emb_filter(
        self, query_list: List[str], num_selected: int
    ) -> Tuple[List[List[float]], List[List[Entry]]]:
        """embedding search

        `This function is profiled as "database.FaissSearch.emb_filter.profile".`

        :param query_list: list of queries
        :type query_list: List[str]
        :param num_selected: number of returned
        :type num_selected: int
        :return: distances and list of entries
        :rtype: Tuple[List[List[float]], List[List[Entry]]]
        """
        #print("FAISS QUERY : ", query_list)
        if type(query_list[0]) == str:
            query_list = self.model.encode(query_list)
        assert type(query_list[0]) == np.ndarray
        D_list, para_id_list = self.index_search(query_list, num_selected)
        entry_list = self.para_id_list_to_entry(para_id_list)
        return D_list, entry_list

    @PROFILER.profile_function("database.FaissSearch.index_search.profile")
    def index_search(
        self, query_list: List[str], num_selected: int
    ) -> Tuple[List[List[float]], List[List[int]]]:
        """faiss index search function

        `This function is profiled as "database.FaissSearch.index_search.profile".`

        :param query_list: list of queries
        :type query_list: List[str]
        :param num_selected: number of returned
        :type num_selected: int
        :return: distances and list of entries
        :rtype: Tuple[List[List[float]], List[List[Entry]]]
        """
        if os.path.split(self.index_path)[-1] == "IVFSQ_IP.index":
            faiss.normalize_L2(query_list)
        self.index.nprobe = self.nprobe
        D_list, para_id_list = self.index.search(query_list, num_selected)
        return D_list, para_id_list
