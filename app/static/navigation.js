(() => {
  const setView = (requested, push) => {
    const view = ['dashboard', 'history', 'logs'].includes(requested) ? requested : 'dashboard';
    document.querySelectorAll('[data-view-panel]').forEach(panel => {
      panel.hidden = panel.dataset.viewPanel !== view;
    });
    document.querySelectorAll('[data-view-link]').forEach(link => {
      const active = link.dataset.viewLink === view;
      link.classList.toggle('is-active', active);
      if (active) link.setAttribute('aria-current', 'page');
      else link.removeAttribute('aria-current');
    });
    document.body.dataset.view = view;
    if (push) window.history.pushState({ view }, '', `/?view=${view}`);
    window.dispatchEvent(new CustomEvent('autoace:viewchange', { detail: { view } }));
  };

  document.addEventListener('click', event => {
    const link = event.target.closest('[data-view-link]');
    if (!link) return;
    event.preventDefault();
    setView(link.dataset.viewLink, true);
  });
  window.addEventListener('popstate', () => setView(new URLSearchParams(location.search).get('view'), false));
  window.addEventListener('DOMContentLoaded', () => setView(new URLSearchParams(location.search).get('view'), false));
})();
