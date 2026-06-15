"""Platform-specific screen-flow extractors."""

from klit_flow.flows.base import ScreenFlowExtractor


def get_extractor(platform: str) -> ScreenFlowExtractor:
    """Return the ScreenFlowExtractor for *platform*.

    Only the Android extractor is fully implemented in Phase 5.
    The others are stubs that return empty results.
    """
    if platform == "android":
        from klit_flow.flows.android import AndroidFlowExtractor

        return AndroidFlowExtractor()
    if platform == "ios":
        from klit_flow.flows.ios import IOSFlowExtractor

        return IOSFlowExtractor()
    if platform == "react_native":
        from klit_flow.flows.react_native import ReactNativeFlowExtractor

        return ReactNativeFlowExtractor()
    if platform == "flutter":
        from klit_flow.flows.flutter import FlutterFlowExtractor

        return FlutterFlowExtractor()
    raise ValueError(f"Unknown platform: {platform!r}")
