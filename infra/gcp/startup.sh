#!/usr/bin/env bash
# =============================================================================
# Startup script executado pela VM GCP no primeiro boot.
# - Instala Docker
# - Monta disco de dados em /opt/mlops
# - Clona o repositório
# - Sobe a stack via docker compose
# =============================================================================
set -euxo pipefail

LOGFILE=/var/log/mlops-startup.log
exec > >(tee -a "$LOGFILE") 2>&1

# ----- 1. Pacotes base
apt-get update
apt-get install -y \
  ca-certificates curl gnupg lsb-release git make jq xfsprogs

# ----- 2. Docker Engine + Compose plugin
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
  gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | \
  tee /etc/apt/sources.list.d/docker.list > /dev/null

apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

systemctl enable --now docker

# ----- 3. Monta disco de dados (device-name=data)
DEVICE=/dev/disk/by-id/google-data
MOUNT=/opt/mlops

mkdir -p "$MOUNT"

if ! blkid "$DEVICE" >/dev/null 2>&1; then
  mkfs.ext4 -F -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard "$DEVICE"
fi

UUID=$(blkid -s UUID -o value "$DEVICE")
grep -q "$UUID" /etc/fstab || \
  echo "UUID=$UUID $MOUNT ext4 discard,defaults,nofail 0 2" >> /etc/fstab

mount -a
mkdir -p $MOUNT/volumes/{postgres,minio}
mkdir -p $MOUNT/logs

# ----- 4. Clona o repositório
# AJUSTE: substitua pela URL do seu repo (HTTPS ou via Deploy Key).
REPO_URL="${REPO_URL:-https://github.com/andreinsardi/mlops-pratica.git}"
cd /opt/mlops
if [ ! -d /opt/mlops/app ]; then
  git clone "$REPO_URL" app
fi
cd /opt/mlops/app

# ----- 5. .env
[ -f .env ] || cp .env.example .env

# ----- 6. Sobe a stack com override de volumes para o disco montado
docker compose \
  -f docker-compose.yml \
  -f infra/gcp/docker-compose.override.yml \
  up -d

echo "✅ Startup concluído."
