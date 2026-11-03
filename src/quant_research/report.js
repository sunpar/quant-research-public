(() => {
  const report = document.getElementById("report");
  const disclosures = new Map();

  function expand(body, expanded) {
    body.classList.toggle("folded", !expanded);
    const button = disclosures.get(body);
    const label = body.dataset.appendix === "exposure" ? "exposure evidence" : "artifact list";
    button.textContent = `${expanded ? "Hide" : "Show"} ${label}`;
    button.setAttribute("aria-expanded", String(expanded));
  }

  for (const body of report.querySelectorAll(".appendix")) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "disclosure";
    button.setAttribute("aria-controls", body.id);
    disclosures.set(body, button);
    body.before(button);
    expand(body, false);
    button.addEventListener("click", () => expand(body, body.classList.contains("folded")));
  }

  function reveal(element) {
    const body = element.closest(".appendix");
    if (body) expand(body, true);
  }

  function followFragment() {
    const target = document.getElementById(window.location.hash.slice(1));
    if (target && report.contains(target)) {
      reveal(target);
      target.focus({ preventScroll: true });
      target.scrollIntoView({ block: "start" });
    }
  }

  window.addEventListener("hashchange", followFragment);
  document.querySelector("nav").addEventListener("click", (event) => {
    const link = event.target.closest("a");
    if (link) {
      const target = document.getElementById(link.hash.slice(1));
      reveal(target);
      target.focus({ preventScroll: true });
    }
  });
  followFragment();

  function element(tag, text = "") {
    const node = document.createElement(tag);
    node.textContent = text;
    return node;
  }

  const form = element("form");
  const label = element("label", "Search evidence");
  label.htmlFor = "evidence-search";
  const input = element("input");
  input.id = "evidence-search";
  input.type = "search";
  input.autocomplete = "off";
  input.setAttribute("aria-describedby", "search-help search-status");
  const help = element("p", "Search includes closed sections. Clear keeps revealed evidence open.");
  help.id = "search-help";
  const controls = element("div");
  controls.className = "search-actions";
  const previous = element("button", "Previous");
  const next = element("button", "Next");
  const clear = element("button", "Clear search");
  for (const button of [previous, next, clear]) button.type = "button";
  const status = element("p");
  status.id = "search-status";
  status.setAttribute("role", "status");
  status.setAttribute("aria-live", "polite");
  controls.append(previous, next, clear);
  form.append(label, input, help, controls, status);
  document.getElementById("search-tools").append(form);

  const blocks = Array.from(report.querySelectorAll("h1, h2, h3, h4, p, li"))
    .filter((block) => !block.parentElement.closest("li"));
  let matches = [];
  let current = -1;

  function search() {
    for (const block of matches) block.classList.remove("match", "current-match");
    const query = input.value.toLowerCase();
    matches = query ? blocks.filter((block) => block.textContent.toLowerCase().includes(query)) : [];
    current = -1;
    for (const block of matches) block.classList.add("match");
    for (const body of disclosures.keys()) {
      if (body.querySelector(".match")) expand(body, true);
    }
    previous.disabled = next.disabled = matches.length === 0;
    clear.disabled = query.length === 0;
    status.textContent = query
      ? (matches.length ? `${matches.length} matching ${matches.length === 1 ? "block" : "blocks"}.`
        : "No matching blocks. Try another word or identifier.")
      : "Enter text to find matching evidence.";
  }

  function select(step) {
    if (matches.length === 0) return;
    if (current >= 0) matches[current].classList.remove("current-match");
    current = current < 0 ? (step > 0 ? 0 : matches.length - 1)
      : (current + step + matches.length) % matches.length;
    const block = matches[current];
    reveal(block);
    block.classList.add("current-match");
    block.tabIndex = -1;
    block.focus({ preventScroll: true });
    block.scrollIntoView({ block: "center" });
    status.textContent = `Match ${current + 1} of ${matches.length} matching blocks.`;
  }

  input.addEventListener("input", search);
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      select(event.shiftKey ? -1 : 1);
    }
  });
  form.addEventListener("submit", (event) => event.preventDefault());
  previous.addEventListener("click", () => select(-1));
  next.addEventListener("click", () => select(1));
  clear.addEventListener("click", () => {
    input.value = "";
    search();
    input.focus();
  });
  search();
})();
