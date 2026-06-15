"""Phase 5 acceptance tests: Android screen-flow extraction."""

from pathlib import Path

import frontmatter

from klit_flow.emit.markdown_emitter import emit_screen_docs
from klit_flow.emit.mermaid_emitter import emit_flow_diagram
from klit_flow.flows import get_extractor
from klit_flow.flows.android import AndroidFlowExtractor
from klit_flow.graph.model import NodeKind, RelationType
from klit_flow.parsing.extractor import extract
from klit_flow.walker import walk

FIXTURE_ROOT = Path(__file__).parent.parent / "fixtures" / "mini_app"


def _setup():
    source_files = walk(FIXTURE_ROOT)
    symbols_by_file = {str(sf.path): extract(sf.path, sf.language) for sf in source_files}
    extractor = AndroidFlowExtractor()
    screens = extractor.extract_screens(source_files, symbols_by_file)
    flows = extractor.extract_flows(source_files, symbols_by_file, screens)
    return source_files, symbols_by_file, screens, flows


# ── Screen detection ──────────────────────────────────────────────────────────


def test_three_screens_detected():
    _, _, screens, _ = _setup()
    names = {s.name for s in screens}
    assert "MainActivity" in names
    assert "AuthActivity" in names
    assert "ProfileActivity" in names


def test_screens_are_screen_kind():
    _, _, screens, _ = _setup()
    assert all(s.kind == NodeKind.Screen for s in screens)


def test_repository_not_a_screen():
    _, _, screens, _ = _setup()
    assert "Repository" not in {s.name for s in screens}


def test_utils_not_a_screen():
    _, _, screens, _ = _setup()
    assert "Utils" not in {s.name for s in screens}


def test_screen_ids_are_unique():
    _, _, screens, _ = _setup()
    ids = [s.id for s in screens]
    assert len(ids) == len(set(ids))


def test_screen_line_numbers_are_positive():
    _, _, screens, _ = _setup()
    for s in screens:
        assert s.start_line >= 1
        assert s.end_line >= s.start_line


# ── Navigation edges ──────────────────────────────────────────────────────────


def test_navigates_to_edges_exist():
    _, _, _, flows = _setup()
    nav = [e for e in flows if e.type == RelationType.NAVIGATES_TO]
    assert len(nav) >= 2  # main→auth and auth→profile


def test_main_navigates_to_auth():
    _, _, screens, flows = _setup()
    screen_by_name = {s.name: s for s in screens}
    nav = [e for e in flows if e.type == RelationType.NAVIGATES_TO]
    assert any(
        e.src_id == screen_by_name["MainActivity"].id
        and e.dst_id == screen_by_name["AuthActivity"].id
        for e in nav
    )


def test_auth_navigates_to_profile():
    _, _, screens, flows = _setup()
    screen_by_name = {s.name: s for s in screens}
    nav = [e for e in flows if e.type == RelationType.NAVIGATES_TO]
    assert any(
        e.src_id == screen_by_name["AuthActivity"].id
        and e.dst_id == screen_by_name["ProfileActivity"].id
        for e in nav
    )


def test_nav_edges_have_honest_confidence():
    _, _, _, flows = _setup()
    for e in flows:
        if e.type == RelationType.NAVIGATES_TO:
            assert 0.0 < e.confidence < 1.0, (
                f"Confidence must be < 1 (best-effort), got {e.confidence}"
            )


def test_nav_edges_have_trigger():
    _, _, _, flows = _setup()
    for e in flows:
        if e.type == RelationType.NAVIGATES_TO:
            assert e.trigger in ("button_tap", "programmatic", "deep_link"), (
                f"Unexpected trigger: {e.trigger!r}"
            )


def test_main_to_auth_trigger_is_button_tap():
    """setOnClickListener in MainActivity → trigger should be button_tap."""
    _, _, screens, flows = _setup()
    screen_by_name = {s.name: s for s in screens}
    main_to_auth = next(
        (
            e
            for e in flows
            if e.type == RelationType.NAVIGATES_TO
            and e.src_id == screen_by_name["MainActivity"].id
            and e.dst_id == screen_by_name["AuthActivity"].id
        ),
        None,
    )
    assert main_to_auth is not None
    assert main_to_auth.trigger == "button_tap"


def test_auth_to_profile_trigger_is_programmatic():
    """login() has no click listener wrapper → trigger should be programmatic."""
    _, _, screens, flows = _setup()
    screen_by_name = {s.name: s for s in screens}
    auth_to_profile = next(
        (
            e
            for e in flows
            if e.type == RelationType.NAVIGATES_TO
            and e.src_id == screen_by_name["AuthActivity"].id
            and e.dst_id == screen_by_name["ProfileActivity"].id
        ),
        None,
    )
    assert auth_to_profile is not None
    assert auth_to_profile.trigger == "programmatic"


def test_no_self_navigation_edges():
    _, _, _, flows = _setup()
    for e in flows:
        if e.type == RelationType.NAVIGATES_TO:
            assert e.src_id != e.dst_id


def test_edges_are_deduplicated():
    """Code + XML both detect same edges; only one copy per (src, dst) should remain."""
    _, _, _, flows = _setup()
    pairs = [(e.src_id, e.dst_id) for e in flows if e.type == RelationType.NAVIGATES_TO]
    assert len(pairs) == len(set(pairs))


def test_xml_edge_wins_over_code_edge():
    """When both code and XML detect the same edge, XML confidence (0.95) should win."""
    _, _, _, flows = _setup()
    for e in flows:
        if e.type == RelationType.NAVIGATES_TO:
            assert e.confidence == 0.95  # XML confidence wins after de-duplication


# ── Factory ───────────────────────────────────────────────────────────────────


def test_get_extractor_returns_android():
    ext = get_extractor("android")
    assert isinstance(ext, AndroidFlowExtractor)


def test_get_extractor_stubs_return_empty():
    source_files = walk(FIXTURE_ROOT)
    symbols_by_file = {str(sf.path): extract(sf.path, sf.language) for sf in source_files}
    for platform in ("ios", "react_native", "flutter"):
        ext = get_extractor(platform)
        assert ext.extract_screens(source_files, symbols_by_file) == []
        assert ext.extract_flows(source_files, symbols_by_file, []) == []


# ── Markdown screen docs ──────────────────────────────────────────────────────


def test_screen_docs_written(tmp_path: Path):
    _, _, screens, flows = _setup()
    paths = emit_screen_docs(screens, flows, tmp_path)
    assert len(paths) == len(screens)
    for p in paths:
        assert p.exists() and p.suffix == ".md"


def test_screen_docs_in_correct_directory(tmp_path: Path):
    _, _, screens, flows = _setup()
    paths = emit_screen_docs(screens, flows, tmp_path)
    expected = tmp_path / "docs" / "screens"
    for p in paths:
        assert p.parent == expected


def test_screen_doc_frontmatter_fields(tmp_path: Path):
    _, _, screens, flows = _setup()
    paths = emit_screen_docs(screens, flows, tmp_path)
    required = {"id", "kind", "path", "reachable_from", "navigates_to"}
    for p in paths:
        post = frontmatter.load(str(p))
        assert required <= post.metadata.keys()
        assert post["kind"] == "Screen"
        assert isinstance(post["reachable_from"], list)
        assert isinstance(post["navigates_to"], list)


def test_screen_doc_main_navigates_to_auth(tmp_path: Path):
    _, _, screens, flows = _setup()
    paths = emit_screen_docs(screens, flows, tmp_path)
    screen_by_name = {s.name: s for s in screens}
    main_doc = next((p for p in paths if "MainActivity" in p.name), None)
    assert main_doc
    post = frontmatter.load(str(main_doc))
    assert screen_by_name["AuthActivity"].id in post["navigates_to"]


def test_screen_doc_empty_for_no_screens(tmp_path: Path):
    paths = emit_screen_docs([], [], tmp_path)
    assert paths == []


# ── Mermaid flow diagram ──────────────────────────────────────────────────────


def test_flow_diagram_written(tmp_path: Path):
    _, _, screens, flows = _setup()
    out = emit_flow_diagram(screens, flows, tmp_path)
    assert out.exists() and out.name == "flows.mmd"


def test_flow_diagram_in_diagrams_directory(tmp_path: Path):
    _, _, screens, flows = _setup()
    out = emit_flow_diagram(screens, flows, tmp_path)
    assert out.parent == tmp_path / "diagrams"


def test_flow_diagram_starts_with_flowchart(tmp_path: Path):
    _, _, screens, flows = _setup()
    out = emit_flow_diagram(screens, flows, tmp_path)
    assert out.read_text().startswith("flowchart")


def test_flow_diagram_contains_arrows(tmp_path: Path):
    _, _, screens, flows = _setup()
    out = emit_flow_diagram(screens, flows, tmp_path)
    assert "-->" in out.read_text()


def test_flow_diagram_references_screen_names(tmp_path: Path):
    _, _, screens, flows = _setup()
    out = emit_flow_diagram(screens, flows, tmp_path)
    content = out.read_text()
    assert "MainActivity" in content
    assert "AuthActivity" in content
    assert "ProfileActivity" in content


def test_flow_diagram_has_trigger_labels(tmp_path: Path):
    _, _, screens, flows = _setup()
    out = emit_flow_diagram(screens, flows, tmp_path)
    content = out.read_text()
    # button_tap and/or programmatic should appear as edge labels
    assert "button_tap" in content or "programmatic" in content
