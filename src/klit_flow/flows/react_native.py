"""React Native / TypeScript screen-flow extractor — stub (Phase 5 target: Android)."""

from klit_flow.flows.base import ScreenFlowExtractor
from klit_flow.graph.model import GraphEdge, GraphNode
from klit_flow.parsing.extractor import Symbol
from klit_flow.walker import SourceFile


class ReactNativeFlowExtractor(ScreenFlowExtractor):
    def extract_screens(
        self,
        source_files: list[SourceFile],
        symbols_by_file: dict[str, list[Symbol]],
    ) -> list[GraphNode]:
        return []

    def extract_flows(
        self,
        source_files: list[SourceFile],
        symbols_by_file: dict[str, list[Symbol]],
        screen_nodes: list[GraphNode],
    ) -> list[GraphEdge]:
        return []
