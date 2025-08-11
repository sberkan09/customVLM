#!/usr/bin/env bash
set -euo pipefail

# --- Settings ---
# Where the persistent key should live (override with SSH_PERSISTENT_KEY if you like)
KEY_PATH="${SSH_PERSISTENT_KEY:-/workspace/.ssh/id_ed25519}"
# Email comment for new keys (override with SSH_KEY_EMAIL)
KEY_EMAIL="${SSH_KEY_EMAIL:-samiakkaya396@gmail.com}"

# --- Ensure ~/.ssh exists with safe perms ---
mkdir -p "${HOME}/.ssh"
chmod 700 "${HOME}/.ssh"

# --- Generate key if missing (no passphrase for automation; add -N "" if you want a passphrase) ---
if [[ ! -f "${KEY_PATH}" ]]; then
  echo "No key at ${KEY_PATH}; generating a new ed25519 key..."
  ssh-keygen -t ed25519 -C "${KEY_EMAIL}" -f "${KEY_PATH}" -N ""
fi

# --- Permissions ---
chmod 600 "${KEY_PATH}"
if [[ -f "${KEY_PATH}.pub" ]]; then chmod 644 "${KEY_PATH}.pub"; fi

# --- Write ~/.ssh/config entries (append once) ---
CONFIG_FILE="${HOME}/.ssh/config"
touch "${CONFIG_FILE}"
chmod 600 "${CONFIG_FILE}"

add_host_block () {
  local host="$1"
  local hostname="$2"
  local identity="$3"

  # Only add if not already present for this host
  if ! grep -qiE "^Host[[:space:]]+${host}([[:space:]]|$)" "${CONFIG_FILE}"; then
    {
      echo ""
      echo "Host ${host}"
      echo "    HostName ${hostname}"
      echo "    User git"
      echo "    IdentityFile ${identity}"
      echo "    IdentitiesOnly yes"
      echo "    AddKeysToAgent yes"
      # Safer default: accept-new avoids MITM prompts on first connect
      echo "    StrictHostKeyChecking accept-new"
    } >> "${CONFIG_FILE}"
  fi
}

add_host_block "github.com" "github.com" "${KEY_PATH}"
add_host_block "gitlab.com" "gitlab.com" "${KEY_PATH}"
add_host_block "bitbucket.org" "bitbucket.org" "${KEY_PATH}"

# --- Seed known_hosts to avoid prompts ---
KNOWN_HOSTS="${HOME}/.ssh/known_hosts"
touch "${KNOWN_HOSTS}"
chmod 644 "${KNOWN_HOSTS}"

# Collect keys (dedupe with sort -u)
{
  ssh-keyscan -T 5 github.com 2>/dev/null || true
  ssh-keyscan -T 5 gitlab.com 2>/dev/null || true
  ssh-keyscan -T 5 bitbucket.org 2>/dev/null || true
} | sort -u | uniq >> "${KNOWN_HOSTS}"

# --- Optionally load into ssh-agent for this session (safe no-op if agent missing) ---
if command -v ssh-agent >/dev/null 2>&1; then
  # Start agent if needed
  if [[ -z "${SSH_AUTH_SOCK:-}" ]]; then
    eval "$(ssh-agent -s)" >/dev/null
  fi
  # Add key if not already loaded
  if ! ssh-add -l 2>/dev/null | grep -q "$(ssh-keygen -lf "${KEY_PATH}" | awk '{print $2}')" ; then
    ssh-add "${KEY_PATH}" >/dev/null || true
  fi
fi

# --- Show where the public key is (handy to copy to GitHub/servers) ---
if [[ -f "${KEY_PATH}.pub" ]]; then
  echo
  echo "Public key (${KEY_PATH}.pub):"
  cat "${KEY_PATH}.pub"
  echo
fi

echo "Persistent SSH key ready at: ${KEY_PATH}"
