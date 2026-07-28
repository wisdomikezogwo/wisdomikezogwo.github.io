/* ============================================================
   Renders an immersive paper walkthrough from papers.js.
   Usage: paper.html?id=<key>
   ============================================================ */

function iconFor(k) {
  return ({ arxiv: "fa-file-lines", pdf: "fa-file-pdf", web: "fa-globe",
            code: "fa-code", journal: "fa-book" })[k] || "fa-link";
}
function labelFor(k) {
  return ({ arxiv: "arXiv", pdf: "PDF", web: "Project page",
            code: "Code", journal: "Journal" })[k] || k;
}

function getId() {
  const p = new URLSearchParams(location.search).get("id");
  return (p && PAPERS[p]) ? p : ORDER[0];
}

function render() {
  const id = getId();
  const d = PAPERS[id];
  document.title = d.title.split(":")[0] + " — Wisdom O. Ikezogwo";

  const links = Object.entries(d.links || {})
    .map(([k, v]) => `<a href="${v}" target="_blank" rel="noopener"><i class="fas ${iconFor(k)}"></i> ${labelFor(k)}</a>`)
    .join("");

  const stats = (d.stats && d.stats.length)
    ? `<div class="section"><div class="stat-row">${d.stats.map(s =>
        `<div class="stat-tile"><div class="stat-value">${s.value}</div><div class="stat-label">${s.label}</div></div>`).join("")}</div></div>`
    : "";

  const beats = (d.beats || []).map(b => `
    <div class="beat" data-kind="${b.kind}">
      <div class="beat-kind">${beatLabel(b.kind)}</div>
      <h2>${b.heading}</h2>
      <p>${b.text}</p>
      ${b.figure ? `<figure class="beat-figure"><img src="${b.figure}" alt="">${b.caption ? `<figcaption>${b.caption}</figcaption>` : ""}</figure>` : ""}
    </div>`).join("");

  const embed = d.embed ? `
    <div class="section">
      <h2 class="section-title">Read the paper</h2>
      <div class="embed-wrap">
        <iframe class="embed-frame" src="${d.embed}" title="Paper PDF" loading="lazy"></iframe>
        <p class="embed-fallback">If the paper doesn't load inline, open it directly:
          <a href="${d.links.arxiv || d.links.pdf || d.links.journal || d.embed}" target="_blank" rel="noopener">view source &rarr;</a></p>
      </div>
    </div>` : "";

  document.getElementById("story").innerHTML = `
    <div class="story-hero">
      <div class="story-venue">${d.venue}${d.award ? ` &middot; <span class="award">${d.award}</span>` : ""}</div>
      <h1>${d.title}</h1>
      <div class="story-authors">${d.authors}</div>
      <div class="story-links">${links}</div>
      <p class="story-tagline">${d.tagline}</p>
    </div>
    ${d.teaser ? `<div class="story-teaser"><img src="${d.teaser}" alt=""></div>` : ""}
    <div class="section" style="border-top:none;padding-top:8px;">${beats}</div>
    ${stats}
    ${embed}
    ${threadNav(id)}
  `;

  observeBeats();
}

function beatLabel(k) {
  return ({ problem: "The problem", insight: "The insight", method: "How it works",
            result: "The result", why: "Why it matters" })[k] || "";
}

function threadNav(id) {
  const i = ORDER.indexOf(id);
  const prev = ORDER[i - 1], next = ORDER[i + 1];
  const cell = (pid, dir) => {
    if (!pid) return `<span></span>`;
    const p = PAPERS[pid];
    return `<a class="${dir === "next" ? "next" : ""}" href="paper.html?id=${pid}">
      <div class="dir">${dir === "next" ? "Next in the thread &rarr;" : "&larr; Earlier in the thread"}</div>
      <div class="tn-title">${p.title.split(":")[0]}</div></a>`;
  };
  return `<div class="section"><div class="thread-nav">${cell(prev, "prev")}${cell(next, "next")}</div></div>`;
}

function observeBeats() {
  const beats = document.querySelectorAll(".beat");
  const io = new IntersectionObserver((entries) => {
    entries.forEach(en => { if (en.isIntersecting) { en.target.classList.add("in"); io.unobserve(en.target); } });
  }, { threshold: 0.15 });
  beats.forEach(b => io.observe(b));

  const bar = document.getElementById("progressBar");
  const onScroll = () => {
    const h = document.documentElement;
    const pct = (h.scrollTop) / (h.scrollHeight - h.clientHeight || 1);
    bar.style.width = Math.min(100, Math.max(0, pct * 100)) + "%";
  };
  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();
}

render();
