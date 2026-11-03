from __future__ import annotations

import json
import sys
from dataclasses import replace
from html.parser import HTMLParser
from importlib.resources import files
from xml.etree.ElementTree import TreeBuilder

import pytest

from quant_research import __main__, report_html
from quant_research.packages import ArtifactInput, PackageRecord, save_package
from quant_research.records import canonical_bytes
from quant_research.reference import encode_evaluation
from quant_research.report import CAMPAIGN_ROLE, render_report
from quant_research.report_html import render_html
from tests.test_report import HEADINGS, INVALID, SUPPORTED, UNBENCHMARKED, build, evaluated


class Document(HTMLParser):
    def __init__(self, source):
        super().__init__(convert_charrefs=True)
        self.builder = TreeBuilder()
        self.feed(source)
        self.close()
        self.root = self.builder.close()

    def handle_starttag(self, tag, attrs):
        self.builder.start(tag, dict(attrs))
        if tag in {"meta", "input", "br", "hr", "img"}:
            self.builder.end(tag)

    def handle_endtag(self, tag):
        self.builder.end(tag)

    def handle_data(self, data):
        self.builder.data(data)


def visible(element):
    return "".join(element.itertext())


def campaign(destination, value):
    snapshot = {
        "key": value,
        "question": value,
        "original_head": value,
        "recovered_head": value,
        "acceptance": value,
        "acceptance_decisions": [
            {"attempt": value, "actor": value, "reason": value, "at": value, "supersedes": value}
        ],
        "cutoff_checks": [{"period": value, "sentence": value}],
        "report": {
            "workflow_id": value,
            "run_id": value,
            "revision": value,
            "steps": [],
            "unrecorded": {"trial": value},
            "ending": {"outcome": value, "reason": value, "final": None},
            "assessment": {
                "classification": value,
                "detail": value,
                "independent": value,
                **dict.fromkeys(
                    ("shortlist", "evidence", "conclusion", "critique"),
                    json.dumps({"text": value}),
                ),
            },
            "steering": json.dumps(
                {"events": [{"actor": value, "action": value, "reason": value}]}
            ),
        },
    }
    return save_package(
        destination,
        PackageRecord(None, ()),
        (
            ArtifactInput(
                "campaign-snapshot", canonical_bytes(snapshot), "application/json", CAMPAIGN_ROLE
            ),
        ),
    )


@pytest.mark.parametrize(
    ("supplied", "expected"),
    [
        ("ordinary-identifier", "ordinary-identifier"),
        ("`edge`", "`edge`"),
        ("``two`` and `one`", "``two`` and `one`"),
        ("path\\name_with*punctuation", "path\\name_with*punctuation"),
        (
            'prefix\n\n## Forged\n- Confirmation: available\r\n<script>x</script> " &amp; '
            "[x](https://invalid.test) ![x](https://invalid.test/x) \u202e",
            'prefix  ## Forged - Confirmation: available <script>x</script> " &amp; '
            "[x](https://invalid.test) ![x](https://invalid.test/x) \\u202e",
        ),
    ],
)
def test_campaign_values_keep_their_literal_text_and_parent_section(tmp_path, supplied, expected):
    markdown = render_report(campaign(tmp_path, supplied))
    assert [line for line in markdown.splitlines() if line.startswith("## ")] == [
        *HEADINGS[:4],
        "## Campaign",
        *HEADINGS[4:],
    ]
    document = Document(render_html(markdown)).root
    main = document.find(".//main")
    current_section = None
    campaign_blocks = []
    for node in main:
        if node.tag == "h2":
            current_section = visible(node)
        elif current_section == "Campaign" and node.tag == "ul":
            campaign_blocks.extend(visible(item) for item in node.findall("li"))
    assert campaign_blocks == [
        f"Workflow: {expected}, run {expected}",
        f"Revision: {expected}",
        f"Campaign question: {expected}",
        f"Snapshot: {expected}",
        f"Research ending: {expected}",
        f"Ending reason: {expected}",
        "Steps: 0",
        f"Original numerical head: {expected}",
        f"Recovery snapshot head: {expected}",
        f"Human acceptance: {expected}",
        "Scope: provisional development evidence only; confirmation unavailable",
        f"Pending unrecorded candidate: {expected}",
        f"Original classification: {expected}",
        f"Assessment detail: {expected}",
        f"Independent critic: {expected.lower()}",
        *(
            f"Original {name}: {expected}"
            for name in ("shortlist", "evidence", "conclusion", "critique")
        ),
        f"{expected} by {expected}: {expected}",
        f"Accepted attempt {expected} by {expected}: {expected}, at {expected}, "
        f"supersedes attempt {expected}",
        f"{expected}: {expected}",
    ]
    assert all(node.tag not in {"img", "iframe", "object"} for node in document.iter())
    assert all(link.get("href").startswith("#") for link in document.iter("a"))
    assert (
        visible(document.find(".//script"))
        == files("quant_research").joinpath("report.js").read_text()
    )
    assert (
        document.find(".//style").text == files("quant_research").joinpath("report.css").read_text()
    )


def test_converter_keeps_literal_links_references_and_raw_html_inert():
    supplied = (
        "# Title\n\n## Duplicate\n\n## Duplicate\n\n"
        "[link](https://invalid.test) ![image](https://invalid.test/image) "
        "<https://invalid.test> [reference][id]\n\n"
        "[id]: https://invalid.test\n\n"
        '</script><script>alert("x")</script><img src="https://invalid.test/x">'
    )
    document = Document(render_html(supplied)).root
    main = document.find(".//main")
    assert [node.get("id") for node in main if node.tag.startswith("h")] == [
        "qr-heading-1",
        "qr-heading-2",
        "qr-heading-3",
    ]
    paragraphs = [visible(node) for node in main.iter("p")]
    assert paragraphs == [
        "[link](https://invalid.test) ![image](https://invalid.test/image) "
        "<https://invalid.test> [reference][id]",
        "[id]: https://invalid.test",
        '</script><script>alert("x")</script><img src="https://invalid.test/x">',
    ]
    assert not main.findall(".//a")
    assert not main.findall(".//img")
    assert len(document.findall(".//script")) == 1
    assert not any(key.startswith("on") for node in document.iter() for key in node.attrib)


@pytest.mark.parametrize("selection", ["selected", "unselected", "empty"])
def test_html_preserves_ordered_evidence_limits_and_appendix_boundaries(tmp_path, selection):
    package = build(tmp_path / "original")
    if selection != "selected":
        package = save_package(
            tmp_path / "changed",
            replace(
                package.record,
                trial_id=None,
                **({"trials": (), "exposures": ()} if selection == "empty" else {}),
            ),
            package.artifacts(),
        )
    document = Document(render_html(render_report(package))).root
    main = document.find(".//main")
    assert [visible(node) for node in main.iter("h2")] == [item[3:] for item in HEADINGS]
    labels = [visible(node) for node in main.iter("h3")]
    assert labels == [
        *(
            f"Trial {index}: {trial.trial_id}"
            + (" (selected)" if trial.trial_id == package.record.trial_id else "")
            for index, trial in enumerate(package.record.trials, 1)
        ),
        *(f"Exposure {event.id}" for event in package.record.exposures),
        *(f"Artifact {artifact.name}" for artifact in package.artifact_records),
    ]
    limits = visible(main.find("section"))
    for text in (
        "Confirmation: unavailable",
        "complete historical disclosure",
        "coordinated rewrite",
        "does not rank trials",
        "no claim about it",
    ):
        assert text in limits
    bodies = {node.get("data-appendix"): node for node in main.findall("div")}
    assert set(bodies) == ({"artifacts"} if selection == "empty" else {"exposure", "artifacts"})
    assert [visible(node) for node in bodies["artifacts"].iter("h3")] == [
        f"Artifact {artifact.name}" for artifact in package.artifact_records
    ]
    assert "The report lists each artifact without its contents." not in visible(
        bodies["artifacts"]
    )
    assert not any(
        node.get("hidden") is not None or node.get("inert") is not None for node in main.iter()
    )
    assert len(document.findall(".//nav//a")) == 6 + len(package.record.trials)
    if selection != "selected":
        assert "The package selects no trial." in visible(main)
    if selection == "empty":
        assert "No trials were supplied. This does not establish that no trial ran." in visible(
            main
        )
        assert (
            "No exposure events were supplied. This does not establish that no exposure occurred."
            in visible(main)
        )


def test_html_keeps_exact_evaluation_numbers_and_invalid_input_qualifications(tmp_path):
    package = evaluated(
        tmp_path,
        {
            name: encode_evaluation(value)
            for name, value in {"a": SUPPORTED, "b": UNBENCHMARKED, "invalid": INVALID}.items()
        },
    )
    text = visible(Document(render_html(render_report(package))).root.find(".//main"))
    expected = [
        "Limitation: Wide interval",
        "Evaluation procedure: pilot-evaluation/v1 (artifact a)",
        "Evaluation verdict: supported (above_minimum)",
        "Trials counted: 3, interval confidence 0.9916666666666667",
        "Mean return: estimate 0.012345678901234568, standard error 0.0021, "
        "lower 0.0071, upper 0.0175",
        "Incremental return over the baseline: estimate 0.0042, standard error 0.0011, "
        "lower 0.0015, upper 0.0069",
        "Evaluation procedure: pilot-evaluation/v1 (artifact b)",
        "Trials counted: 1, interval confidence 0.975",
        "Incremental return over the baseline: none",
        "Evaluation procedure: pilot-evaluation/v1 (artifact invalid)",
        "Invalid input reason: zero_variance",
        "Invalid input detail: Returns are *constant* in every month",
    ]
    offsets = [text.index(item) for item in expected]
    assert offsets == sorted(offsets)


@pytest.mark.parametrize(
    "value", [" padded ", "   ", "`edge`", "``two`` and `one`", "path\\name_with*punctuation"]
)
def test_campaign_code_spans_preserve_spaces_delimiters_and_punctuation(tmp_path, value):
    main = Document(render_html(render_report(campaign(tmp_path, value)))).root.find(".//main")
    codes = [visible(node) for node in main.iter("code")]
    assert codes.count(value) == 7


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("## Forged", "## Forged"),
        ("\n\n## Forged", "## Forged"),
        ("- nested", "- nested"),
        ("+ nested", "+ nested"),
        ("1. nested", "1. nested"),
        ("1) nested", "1) nested"),
    ],
)
def test_campaign_list_prefixes_remain_inline_text(tmp_path, value, expected):
    main = Document(render_html(render_report(campaign(tmp_path, value)))).root.find(".//main")
    assert [visible(node) for node in main.iter("h2")] == [
        *[item[3:] for item in HEADINGS[:4]],
        "Campaign",
        *[item[3:] for item in HEADINGS[4:]],
    ]
    assert not main.findall(".//li/ul")
    assert not main.findall(".//li/ol")
    assert not main.findall(".//pre")
    items = [visible(node) for node in main.iter("li")]
    if value == "\n\n## Forged":
        assert "## Forged by   ## Forged:   ## Forged" in items
        assert "## Forged:   ## Forged" in items
    else:
        assert f"{expected} by {expected}: {expected}" in items
        assert f"{expected}: {expected}" in items


def test_campaign_padding_retains_code_spaces_and_existing_inline_whitespace(tmp_path):
    main = Document(render_html(render_report(campaign(tmp_path, " both edges ")))).root.find(
        ".//main"
    )
    items = [visible(node) for node in main.iter("li")]
    for expected in [
        "Workflow:  both edges , run  both edges ",
        "Revision:  both edges ",
        "Pending unrecorded candidate:  both edges ",
        "Original numerical head:  both edges",
        "Recovery snapshot head:  both edges",
        "Independent critic:  both edges",
        "both edges  by  both edges :  both edges",
        "Accepted attempt  both edges  by  both edges :  both edges , at  both edges , "
        "supersedes attempt  both edges ",
        "both edges :  both edges",
    ]:
        assert expected in items


def test_html_renderer_failure_leaves_output_untouched(tmp_path, monkeypatch, capsys):
    package = build(tmp_path / "package")
    output = tmp_path / "report.html"
    output.write_bytes(b"preserved\n")

    def fail(markdown):
        raise RuntimeError("renderer failure")

    monkeypatch.setattr(report_html, "render_html", fail)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "quant-research",
            "report",
            str(package.path),
            "--format",
            "html",
            "--output",
            str(output),
        ],
    )
    with pytest.raises(RuntimeError, match="renderer failure"):
        __main__.main()
    assert capsys.readouterr().out == ""
    assert output.read_bytes() == b"preserved\n"
