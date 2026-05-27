#!/usr/bin/env bash
# =============================================================================
# Provisiona a VM GCP para o projeto MLOps Prática.
# Pré-requisitos: gcloud autenticado, GCP_PROJECT setado.
# Uso: ./infra/gcp/provision.sh
# =============================================================================
set -euo pipefail

: "${GCP_PROJECT:?defina GCP_PROJECT}"
GCP_REGION="${GCP_REGION:-southamerica-east1}"
GCP_ZONE="${GCP_ZONE:-southamerica-east1-a}"
VM_NAME="${VM_NAME:-mlops-pratica}"
VM_MACHINE="${VM_MACHINE:-e2-standard-4}"
VM_DISK_SIZE="${VM_DISK_SIZE:-100GB}"
VM_DATA_DISK="${VM_DATA_DISK:-mlops-data}"
VM_DATA_DISK_SIZE="${VM_DATA_DISK_SIZE:-200GB}"
SOURCE_IP_RANGE="${SOURCE_IP_RANGE:-0.0.0.0/0}"   # AJUSTE para seu IP em prod!

echo "==> Definindo projeto: $GCP_PROJECT"
gcloud config set project "$GCP_PROJECT"

echo "==> Habilitando APIs necessárias"
gcloud services enable compute.googleapis.com

echo "==> Criando disco de dados ($VM_DATA_DISK_SIZE pd-ssd)"
gcloud compute disks create "$VM_DATA_DISK" \
  --type=pd-ssd \
  --size="$VM_DATA_DISK_SIZE" \
  --zone="$GCP_ZONE" || echo "Disco já existe — ok"

echo "==> Criando VM $VM_NAME ($VM_MACHINE) na zona $GCP_ZONE"
gcloud compute instances create "$VM_NAME" \
  --zone="$GCP_ZONE" \
  --machine-type="$VM_MACHINE" \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size="$VM_DISK_SIZE" \
  --boot-disk-type=pd-ssd \
  --disk="name=$VM_DATA_DISK,device-name=data,mode=rw,boot=no" \
  --tags=mlops \
  --metadata-from-file=startup-script="$(dirname "$0")/startup.sh"

echo "==> Regras de firewall (libera Airflow/MLflow/MinIO/FastAPI)"
gcloud compute firewall-rules create allow-mlops-uis \
  --network=default \
  --allow=tcp:8080,tcp:5000,tcp:9001,tcp:8000 \
  --source-ranges="$SOURCE_IP_RANGE" \
  --target-tags=mlops || echo "Regra já existe — ok"

echo ""
echo "==> VM provisionada. Aguardando inicialização (3-5 min)..."
EXTERNAL_IP=$(gcloud compute instances describe "$VM_NAME" \
  --zone="$GCP_ZONE" \
  --format='get(networkInterfaces[0].accessConfigs[0].natIP)')

cat <<EOF

============================================================
✅ VM pronta!
   IP externo: $EXTERNAL_IP

   Aguarde o startup-script terminar (~5 min), depois acesse:
     Airflow:  http://$EXTERNAL_IP:8080  (admin/admin)
     MLflow:   http://$EXTERNAL_IP:5000
     MinIO:    http://$EXTERNAL_IP:9001
     FastAPI:  http://$EXTERNAL_IP:8000/docs

   SSH:
     gcloud compute ssh $VM_NAME --zone=$GCP_ZONE

   Acompanhar startup:
     gcloud compute ssh $VM_NAME --zone=$GCP_ZONE \\
       --command='sudo journalctl -u google-startup-scripts.service -f'

============================================================
EOF
