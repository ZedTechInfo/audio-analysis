((root, module) => {
  const pageCount = (total, pageSize) => Math.max(1, Math.ceil(total / pageSize));
  const clampPage = (page, total, pageSize) => Math.min(Math.max(1, page), pageCount(total, pageSize));
  const pageItems = (items, page, pageSize) => items.slice((clampPage(page, items.length, pageSize) - 1) * pageSize, clampPage(page, items.length, pageSize) * pageSize);
  const api = { pageCount, clampPage, pageItems };
  if (module) module.exports = api;
  else root.AutoAcePagination = api;
})(typeof window === 'undefined' ? globalThis : window, typeof module === 'undefined' ? null : module);
