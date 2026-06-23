function parseInterval(value) {
  const match = /^every\s+(\d+(?:\.\d+)?)s$/.exec(value || "");
  if (!match) return null;
  return Number(match[1]) * 1000;
}

function wirePolling(root = document) {
  root.querySelectorAll("[hx-get][hx-trigger][hx-swap='outerHTML']").forEach((element) => {
    if (element.dataset.polling === "true") return;
    const interval = parseInterval(element.getAttribute("hx-trigger"));
    const url = element.getAttribute("hx-get");
    if (!interval || !url) return;

    element.dataset.polling = "true";
    const timer = window.setInterval(async () => {
      if (!document.body.contains(element)) {
        window.clearInterval(timer);
        return;
      }
      const response = await fetch(url, { headers: { "X-Requested-With": "fetch" } });
      if (!response.ok) return;
      const html = await response.text();
      const template = document.createElement("template");
      template.innerHTML = html.trim();
      const replacement = template.content.firstElementChild;
      if (!replacement) return;
      element.replaceWith(replacement);
      window.clearInterval(timer);
      wirePolling(document);
    }, interval);
  });
}

window.addEventListener("DOMContentLoaded", () => wirePolling(document));
