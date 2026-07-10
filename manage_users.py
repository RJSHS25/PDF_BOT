"""
manage_users.py — admin CLI for GurucoolBOT's interim login system
--------------------------------------------------------------------
Passwords are never stored in plaintext: each one is hashed with
PBKDF2-HMAC-SHA256 and a random per-user salt, written to users.json
next to this script.

This is meant as a stopgap until SSO is wired up — swap it out once
that's approved.

Usage:
    python manage_users.py add <username>       # create or reset a user
    python manage_users.py remove <username>     # delete a user
    python manage_users.py list                  # show all usernames
"""

import argparse
import getpass
import json
import os
import hashlib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USERS_FILE = os.path.join(BASE_DIR, "users.json")


def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def save_users(users):
    with open(USERS_FILE, "w", encoding="utf-8") as fh:
        json.dump(users, fh, indent=2)


def hash_password(password, salt=None):
    salt = salt or os.urandom(16)
    hashed = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000)
    return salt.hex(), hashed.hex()


def add_user(username):
    password = getpass.getpass(f"Password for '{username}': ")
    confirm = getpass.getpass("Confirm password: ")
    if not password:
        print("Password cannot be empty.")
        return
    if password != confirm:
        print("Passwords don't match — try again.")
        return

    salt_hex, hash_hex = hash_password(password)
    users = load_users()
    is_update = username in users
    users[username] = {"salt": salt_hex, "hash": hash_hex}
    save_users(users)
    print(f"User '{username}' {'updated' if is_update else 'created'} successfully.")


def remove_user(username):
    users = load_users()
    if username not in users:
        print(f"No such user: '{username}'")
        return
    del users[username]
    save_users(users)
    print(f"User '{username}' removed.")


def list_users():
    users = load_users()
    if not users:
        print("No users configured yet.")
        return
    for name in sorted(users):
        print(f"- {name}")


def main():
    parser = argparse.ArgumentParser(description="Manage GurucoolBOT user accounts.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_parser = subparsers.add_parser("add", help="Create or reset a user's password")
    add_parser.add_argument("username")

    remove_parser = subparsers.add_parser("remove", help="Delete a user")
    remove_parser.add_argument("username")

    subparsers.add_parser("list", help="List all usernames")

    args = parser.parse_args()

    if args.command == "add":
        add_user(args.username)
    elif args.command == "remove":
        remove_user(args.username)
    elif args.command == "list":
        list_users()


if __name__ == "__main__":
    main()
