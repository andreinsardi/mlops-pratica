# Deploy em VM GCP (Compute Engine)

Guia completo para subir a stack MLOps Prática em uma máquina virtual única do Google Cloud.

## Arquitetura GCP

```
   Internet ──▶ Cloud Load Balancer (opcional)
                       │
                       ▼
         Compute Engine VM (e2-standard-4)
         ┌────────────────────────────────┐
         │ Docker + Docker Compose         │
         │   • Postgres + pgvector        │
         │   • MinIO (S3)                 │
         │   • MLflow                     │
         │   • Airflow webserver+sched    │
         │   • FastAPI                    │
         └────────────────────────────────┘
                 │
         Persistent Disk SSD (100GB)
         (monta /opt/mlops/volumes)
```

## 1. Pré-requisitos

- Projeto GCP com billing habilitado
- `gcloud` CLI instalado e autenticado (`gcloud auth login`)
- Cota suficiente para Compute Engine (>= 4 vCPU, 16 GB RAM)

## 2. Provisionamento via gcloud

Use o script `infra/gcp/provision.sh` (incluso) para automatizar.

### 2.1 Variáveis

```bash
export GCP_PROJECT=meu-projeto-mlops
export GCP_REGION=southamerica-east1
export GCP_ZONE=southamerica-east1-a
export VM_NAME=mlops-pratica
export VM_MACHINE=e2-standard-4         # 4 vCPU, 16 GB RAM
export VM_DISK_SIZE=100GB                # boot disk
export VM_DATA_DISK=mlops-data           # disco adicional
export VM_DATA_DISK_SIZE=200GB
export VM_IMAGE_FAMILY=ubuntu-2204-lts
export VM_IMAGE_PROJECT=ubuntu-os-cloud
```

### 2.2 Comandos `gcloud`

```bash
# Criar disco de dados
gcloud compute disks create $VM_DATA_DISK \
  --project=$GCP_PROJECT \
  --type=pd-ssd \
  --size=$VM_DATA_DISK_SIZE \
  --zone=$GCP_ZONE

# Criar VM com Container-Optimized OS Ubuntu
gcloud compute instances create $VM_NAME \
  --project=$GCP_PROJECT \
  --zone=$GCP_ZONE \
  --machine-type=$VM_MACHINE \
  --image-family=$VM_IMAGE_FAMILY \
  --image-project=$VM_IMAGE_PROJECT \
  --boot-disk-size=$VM_DISK_SIZE \
  --boot-disk-type=pd-ssd \
  --disk=name=$VM_DATA_DISK,device-name=data,mode=rw,boot=no \
  --tags=mlops,http-server,https-server \
  --metadata-from-file=startup-script=infra/gcp/startup.sh

# Regras de firewall (somente IPs de confiança em produção!)
gcloud compute firewall-rules create allow-mlops-uis \
  --project=$GCP_PROJECT \
  --network=default \
  --allow=tcp:8080,tcp:5000,tcp:9001,tcp:8000 \
  --source-ranges=SEU.IP.PUBLICO/32 \
  --target-tags=mlops

# Pegar IP público
gcloud compute instances describe $VM_NAME \
  --zone=$GCP_ZONE \
  --format='get(networkInterfaces[0].accessConfigs[0].natIP)'
```

## 3. Configuração interna da VM (startup-script)

O `infra/gcp/startup.sh` é injetado como `startup-script` da VM e executa:

1. Atualiza pacotes
2. Instala Docker Engine + Compose plugin
3. Monta disco de dados em `/opt/mlops`
4. Clona o repositório (substitua a URL)
5. `cp .env.example .env`
6. `docker compose up -d`

## 4. Acesso às UIs

Após a VM ficar pronta (~3-5 min):

| Serviço | URL |
|---|---|
| Airflow | `http://EXTERNAL_IP:8080` (admin/admin) |
| MLflow | `http://EXTERNAL_IP:5000` |
| MinIO Console | `http://EXTERNAL_IP:9001` |
| FastAPI Swagger | `http://EXTERNAL_IP:8000/docs` |

> **Segurança:** as portas estão abertas para demonstração. Em ambiente real:
> - Use **IAP (Identity-Aware Proxy)** ou **VPN** em vez de abrir IPs.
> - Coloque um **Nginx reverse proxy** com HTTPS (Let's Encrypt via certbot).
> - Troque as credenciais padrão (`admin/admin`, `minioadmin/minioadmin`).

## 5. Diferenças vs. local

| Aspecto | Local | GCP VM |
|---|---|---|
| Persistência | volumes Docker | disco persistente montado em `/opt/mlops/volumes` |
| Boot | manual via `make up` | automático via `startup-script` |
| Backup | manual | snapshot do disk via `gcloud compute disks snapshot` |
| Custo | $0 | ~ US$ 110/mês (e2-standard-4 24/7 + SSD 200GB) |
| TLS | não | recomendado via Nginx + certbot |
| Logs | `docker logs` | `gcloud logging` (se Cloud Ops Agent instalado) |

## 6. docker-compose.gcp.yml (override)

Para a VM, use `infra/gcp/docker-compose.override.yml` que muda os volumes para o disco montado em `/opt/mlops/volumes`:

```bash
docker compose -f docker-compose.yml -f infra/gcp/docker-compose.override.yml up -d
```

## 7. Backup e restore

```bash
# Snapshot diário (configurar via Cloud Scheduler + gcloud)
gcloud compute disks snapshot $VM_DATA_DISK \
  --snapshot-names=mlops-data-$(date +%Y%m%d) \
  --zone=$GCP_ZONE

# Restore
gcloud compute disks create $VM_DATA_DISK-restored \
  --source-snapshot=mlops-data-YYYYMMDD \
  --zone=$GCP_ZONE
```

## 8. Custo estimado (us-east1 / mai-2026)

| Item | Valor mensal (USD) |
|---|---|
| VM e2-standard-4 (24/7) | ~$98 |
| Boot disk pd-ssd 100GB | ~$17 |
| Data disk pd-ssd 200GB | ~$34 |
| Network egress (~10GB/mês) | ~$1 |
| **Total aproximado** | **~$150** |

Dica: use `--preemptible` (interruptíveis) para baixar 70% do custo em demos.

## 9. Limpeza

```bash
gcloud compute instances delete $VM_NAME --zone=$GCP_ZONE
gcloud compute disks delete $VM_DATA_DISK --zone=$GCP_ZONE
gcloud compute firewall-rules delete allow-mlops-uis
```

## 10. Próximos passos (evolução)

- Migrar Postgres para **Cloud SQL** (gerenciado)
- Trocar MinIO por **GCS bucket** (`mlflow --default-artifact-root gs://...`)
- Mover serving para **Cloud Run** (autoscale)
- Airflow gerenciado: **Cloud Composer**
- Monitoramento: **Cloud Monitoring** + **Cloud Logging**
