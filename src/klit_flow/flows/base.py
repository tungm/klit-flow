"""Abstract base class for platform-specific screen-flow extractors."""

from abc import ABC, abstractmethod

from klit_flow.graph.model import GraphEdge, GraphNode
from klit_flow.parsing.extractor import Symbol
from klit_flow.walker import SourceFile


class ScreenFlowExtractor(ABC):
    """Identifies Screen nodes and NAVIGATES_TO edges for one mobile platform."""

    @abstractmethod
    def extract_screens(
        self,
        source_files: list[SourceFile],
        symbols_by_file: dict[str, list[Symbol]],
    ) -> list[GraphNode]:
        """Return Screen-kind GraphNodes for the target app."""

    @abstractmethod
    def extract_flows(
        self,
        source_files: list[SourceFile],
        symbols_by_file: dict[str, list[Symbol]],
        screen_nodes: list[GraphNode],
    ) -> list[GraphEdge]:
        """Return NAVIGATES_TO edges between Screen nodes.

        Edges must carry an honest ``confidence`` score (0–1).
        Unresolvable / dynamic routes must never be silently fabricated.
        """
