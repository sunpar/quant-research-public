from __future__ import annotations

from importlib.resources import files


def render_html(markdown: str) -> str:
    """Convert canonical report Markdown into a self-contained offline HTML document."""
    from markdown_it import MarkdownIt
    from markdown_it.token import Token

    parser = MarkdownIt(
        "commonmark", {"html": False, "linkify": False, "typographer": False}
    ).disable(["link", "image", "autolink", "linkify", "reference"])
    tokens = parser.parse(markdown)
    headings = [index for index, token in enumerate(tokens) if token.type == "heading_open"]
    navigation = []
    section = ""
    for number, index in enumerate(headings, 1):
        heading = tokens[index]
        heading.attrSet("id", f"qr-heading-{number}")
        heading.attrSet("tabindex", "-1")
        if heading.tag == "h2":
            section = tokens[index + 1].content
        if heading.tag == "h2" or (heading.tag == "h3" and section == "Trials"):
            label = parser.renderer.renderInline(tokens[index + 1].children, parser.options, {})
            navigation.append(
                f'<li class="contents-{heading.tag}">'
                f'<a href="#qr-heading-{number}">{label}</a></li>'
            )

    insertions: dict[int, list[str]] = {}
    sections = [index for index in headings if tokens[index].tag == "h2"]
    for start, end in zip(sections, [*sections[1:], len(tokens)], strict=True):
        label = tokens[start + 1].content
        if label == "Limits":
            insertions.setdefault(start, []).append('<section class="limits">\n')
            insertions.setdefault(end, []).append("</section>\n")
        elif label in ("Exposure history", "Artifacts"):
            records = [
                index for index in headings if start < index < end and tokens[index].tag == "h3"
            ]
            if records:
                kind = "exposure" if label == "Exposure history" else "artifacts"
                body_id = tokens[start].attrGet("id").replace("heading", "body")
                insertions.setdefault(records[0], []).append(
                    f'<div class="appendix" id="{body_id}" data-appendix="{kind}">\n'
                )
                insertions.setdefault(end, []).append("</div>\n")
    body = []
    for index in range(len(tokens) + 1):
        for content in insertions.get(index, []):
            body.append(Token("html_block", "", 0, content=content))
        if index < len(tokens):
            body.append(tokens[index])
    evidence = parser.renderer.render(body, parser.options, {})
    resources = files("quant_research")
    css = resources.joinpath("report.css").read_text(encoding="utf-8")
    script = resources.joinpath("report.js").read_text(encoding="utf-8")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Development research report</title>
<style>{css}</style>
</head>
<body>
<a class="skip-link" href="#report">Skip to report</a>
<div class="layout">
<aside class="sidebar">
<nav aria-label="Report contents">
<h2>Contents</h2>
<ul>{"".join(navigation)}</ul>
</nav>
<div id="search-tools"></div>
</aside>
<main id="report" tabindex="-1">
{evidence}</main>
</div>
<script>{script}</script>
</body>
</html>
"""
