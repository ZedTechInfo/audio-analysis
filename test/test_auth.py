import unittest
import uuid

from fastapi.testclient import TestClient

from app import main


class AuthenticationTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main.app)
        self.email = f"tester-{uuid.uuid4().hex[:8]}@example.com"
        self.username = f"tester_{uuid.uuid4().hex[:8]}"
        self.password = "a-secure-password-123"

    def test_public_auth_pages_do_not_render_authenticated_navigation(self):
        for path in ("/login", "/signup", "/forgot-password", "/reset-password"):
            response = self.client.get(path)
            self.assertEqual(response.status_code, 200, path)
            self.assertNotIn('data-view-link="history"', response.text, path)
            self.assertNotIn("Sign out", response.text, path)

    def test_signup_login_and_logout_use_user_session(self):
        signup = self.client.post(
            "/signup",
            data={"email": self.email, "username": self.username, "password": self.password, "confirm_password": self.password},
            follow_redirects=False,
        )
        self.assertEqual(signup.status_code, 303)
        self.assertIn("autoace_session", self.client.cookies)

        logout = self.client.post("/logout", follow_redirects=False)
        self.assertEqual(logout.status_code, 303)
        self.assertNotIn("autoace_session", self.client.cookies)

        login = self.client.post(
            "/login",
            data={"identifier": self.username, "password": self.password},
            follow_redirects=False,
        )
        self.assertEqual(login.status_code, 303)
        self.assertIn("autoace_session", self.client.cookies)

    def test_forgot_password_is_non_enumerating_and_reset_link_works_in_development(self):
        self.client.post(
            "/signup",
            data={"email": self.email, "username": self.username, "password": self.password, "confirm_password": self.password},
        )
        response = self.client.post("/forgot-password", data={"email": self.email})
        self.assertEqual(response.status_code, 200)
        self.assertIn("recovery instructions have been prepared", response.text)
        self.assertIn("/reset-password?token=", response.text)


if __name__ == "__main__":
    unittest.main()
