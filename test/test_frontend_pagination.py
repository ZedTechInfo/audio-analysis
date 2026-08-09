import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class FrontendPaginationContractTests(unittest.TestCase):
    def setUp(self):
        self.dashboard = (ROOT / "app/static/dashboard.js").read_text(encoding="utf-8")
        self.history = (ROOT / "app/templates/partials/history_view.html").read_text(encoding="utf-8")
        self.logs = (ROOT / "app/templates/partials/logs_view.html").read_text(encoding="utf-8")

    def test_history_pagination_has_rendered_controls_and_click_handler(self):
        self.assertIn('id="history-pagination"', self.history)
        self.assertIn('data-history-page', self.dashboard)
        self.assertIn("historyPage = window.AutoAcePagination.clampPage", self.dashboard)
        self.assertIn("$('history-pagination').addEventListener('click'", self.dashboard)

    def test_logs_pagination_has_rendered_controls_and_click_handler(self):
        self.assertIn('id="logs-pagination"', self.logs)
        self.assertIn('data-log-page', self.dashboard)
        self.assertIn("$('logs-pagination').addEventListener('click'", self.dashboard)


if __name__ == "__main__":
    unittest.main()
