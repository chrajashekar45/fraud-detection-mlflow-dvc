# EC2 Deployment Guide

This guide documents how to recreate the EC2 deployment from scratch for the fraud detection MLOps project.

Use this when the current EC2 instance is deleted and a new one needs to be created.

## Current Production Shape

```text
GitHub main branch
    |
    v
GitHub Actions CI
    - install dependencies
    - authenticate to AWS
    - dvc pull model artifacts from S3
    - run pytest
    |
    v
GitHub Actions Deploy
    - SSH into EC2
    - git pull origin main
    - install lightweight DVC dependencies on host
    - dvc pull model artifacts
    - docker-compose up -d --build
    |
    v
EC2 instance
    - FastAPI backend on port 8000
    - Streamlit frontend on port 8501
```

## AWS Resources Used

```text
S3 bucket: fraud-detection-mlflow-dvc
S3 region: ap-southeast-2
IAM user for DVC/S3: dvc-s3-user
EC2 OS: Ubuntu
App ports: 8000 and 8501
SSH user: ubuntu
```

The S3 bucket stores DVC-tracked data and model artifacts under:

```text
s3://fraud-detection-mlflow-dvc/dvc-store
```

## 1. Create EC2 Instance

In AWS EC2 console:

1. Click **Launch instance**.
2. Name:

```text
fraud-detection-mlops-ec2
```

3. AMI:

```text
Ubuntu Server
```

4. Instance type:

```text
t2.micro
```

or another free-tier eligible micro instance.

5. Create or select a key pair:

```text
fraud-detection-ec2-key.pem
```

6. Storage:

```text
20 GB gp3
```

7. Security group inbound rules:

```text
SSH      22    your IP for manual access
Custom   8000  your IP or demo audience IP
Custom   8501  your IP or demo audience IP
```

For GitHub Actions deployment over SSH, GitHub-hosted runners use changing IPs. For the simple portfolio deployment, SSH may need to be temporarily open to:

```text
0.0.0.0/0
```

Only do this if the private key is protected. A safer production option is a self-hosted runner, VPN, SSM Session Manager, or restricted deployment IP range.

## 2. Connect From Local Machine

From local PowerShell:

```powershell
ssh -i path\to\fraud-detection-ec2-key.pem ubuntu@EC2_PUBLIC_IP
```

If SSH complains about key permissions on Windows, fix the key permissions or move the key to a safe folder with restricted access.

## 3. Install System Packages On EC2

Run on EC2:

```bash
sudo apt update
sudo apt install -y git docker.io python3-pip python3-venv curl unzip
sudo systemctl enable docker
sudo systemctl start docker
sudo usermod -aG docker ubuntu
```

Exit and reconnect so the `docker` group membership applies:

```bash
exit
```

Reconnect:

```powershell
ssh -i path\to\fraud-detection-ec2-key.pem ubuntu@EC2_PUBLIC_IP
```

Check Docker:

```bash
docker --version
docker ps
```

If `docker ps` fails with permission denied, reconnect again or temporarily use `sudo docker ps`.

## 4. Install Docker Compose

On the EC2 instance, `docker-compose-plugin` was not available from apt, so Docker Compose was installed manually:

```bash
sudo curl -L "https://github.com/docker/compose/releases/download/v2.29.7/docker-compose-linux-x86_64" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
docker-compose version
```

This project uses:

```bash
docker-compose -f docker/docker-compose.yml up -d --build
```

instead of:

```bash
docker compose -f docker/docker-compose.yml up -d --build
```

because `docker-compose` worked reliably on the EC2 instance.

## 5. Install AWS CLI On EC2

Run:

```bash
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
aws --version
```

Configure credentials:

```bash
aws configure
```

Use the IAM user:

```text
IAM user: dvc-s3-user
Region: ap-southeast-2
Output: json
```

Test S3 access:

```bash
aws s3 ls s3://fraud-detection-mlflow-dvc
```

Expected result: no access error. The bucket may list objects under `dvc-store/`.

## 6. Clone The Repository

On EC2:

```bash
cd ~
git clone https://github.com/YOUR_USERNAME/fraud-detection-mlflow-dvc.git
cd fraud-detection-mlflow-dvc
```

If the folder already exists:

```bash
cd ~/fraud-detection-mlflow-dvc
git pull origin main
```

## 7. Install Only DVC On EC2 Host

Do not install the full `requirements.txt` on the EC2 host.

Reason: the EC2 system Python may be too new, such as Python 3.14. When that happened, `pandas==2.2.3` did not have a matching wheel, so pip tried to compile pandas from source and the micro instance ran out of memory.

The exact failure seen:

```text
cc: fatal error: Killed signal terminated program cc1
metadata-generation-failed
```

The fix is to install only the DVC tooling needed to pull artifacts:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install "dvc[s3]==3.64.2" "pathspec==0.12.1" dvc-s3
```

Why:

- EC2 host only needs DVC to pull artifacts.
- Application dependencies are installed inside the Docker image.
- Dockerfile uses `python:3.12-slim`, which matches the tested project environment better.

## 8. Pull Model Artifacts From DVC/S3

Inside the repository on EC2:

```bash
source .venv/bin/activate
dvc pull models/model.pkl
dvc pull models/pruned_feature_cols.json
```

These files are required before Docker build because the Dockerfile copies them into the image:

```text
models/model.pkl
models/pruned_feature_cols.json
models/eval_report.json
models/latency_report.json
```

If DVC says S3 support is missing:

```text
s3 is supported, but requires 'dvc-s3' to be installed
```

install:

```bash
pip install dvc-s3
```

## 9. Build And Run The App

Run:

```bash
docker-compose -f docker/docker-compose.yml up -d --build
```

Check containers:

```bash
docker ps
```

Expected services:

```text
fraud-api
fraud-ui
```

Check logs if needed:

```bash
docker logs fraud-detection-mlflow-dvc-fraud-api-1
docker logs fraud-detection-mlflow-dvc-fraud-ui-1
```

Container names may vary. Use `docker ps` to copy the actual name.

## 10. Verify The Deployment

From your browser:

```text
http://EC2_PUBLIC_IP:8000/docs
http://EC2_PUBLIC_IP:8501
```

Health check:

```text
http://EC2_PUBLIC_IP:8000/health
```

Expected response:

```json
{
  "status": "ok",
  "model_loaded": true,
  "features": 270,
  "threshold": 0.74
}
```

Streamlit should load and call the backend through Docker Compose using:

```text
FASTAPI_URL=http://fraud-api:8000
```

This is important because inside Docker, `localhost` from the Streamlit container means the Streamlit container itself, not the FastAPI container.

## 11. GitHub Actions Secrets

Repository secrets used by CI/CD:

```text
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
EC2_HOST
EC2_USERNAME
EC2_SSH_KEY
```

Values:

```text
AWS_ACCESS_KEY_ID      IAM access key for dvc-s3-user
AWS_SECRET_ACCESS_KEY  IAM secret key for dvc-s3-user
EC2_HOST               EC2 public IP or public DNS
EC2_USERNAME           ubuntu
EC2_SSH_KEY            full private key contents from the .pem file
```

Do not commit these values to Git.

The CI workflow:

```text
.github/workflows/ci.yml
```

does:

- install dependencies
- configure AWS credentials
- verify AWS identity
- pull DVC model artifacts
- run tests

The deployment workflow:

```text
.github/workflows/deploy.yml
```

does:

- wait for CI success
- SSH into EC2
- run `git pull origin main`
- install lightweight DVC dependencies
- pull model artifacts from S3
- rebuild and restart Docker Compose

## 12. Issues Faced And Fixes

### AWS CLI not recognized on local Windows

Problem:

```text
aws is not recognized
```

Fix:

- close and reopen PowerShell after installing AWS CLI
- or add this to PATH:

```text
C:\Program Files\Amazon\AWSCLIV2\
```

Temporary direct command:

```powershell
& "C:\Program Files\Amazon\AWSCLIV2\aws.exe" --version
```

### DVC push failed because S3 plugin was missing

Problem:

```text
s3 is supported, but requires 'dvc-s3' to be installed
```

Fix:

```bash
pip install dvc-s3
```

or:

```bash
pip install "dvc[s3]"
```

### GitHub Actions region was empty

Problem:

```text
Invalid endpoint: https://s3..amazonaws.com
```

Cause:

```text
AWS_DEFAULT_REGION was empty
```

Fix:

Set region directly in workflow:

```yaml
AWS_DEFAULT_REGION: ap-southeast-2
AWS_REGION: ap-southeast-2
```

### GitHub Actions credentials were not passed

Problem:

AWS variables appeared empty in logs.

Fix:

- add secrets under GitHub repository secrets, not Codespaces or Dependabot secrets
- use exact names:

```text
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
```

- use `aws-actions/configure-aws-credentials@v4`

### Old AWS access key caused DVC pull failure

Problem:

```text
HeadObject operation: 400 Bad Request
```

Fix:

- create a new IAM access key
- update GitHub repository secrets
- rerun workflow

### `docker-compose-plugin` package not found

Problem:

```text
Unable to locate package docker-compose-plugin
```

Fix:

Install standalone `docker-compose` binary manually and use:

```bash
docker-compose -f docker/docker-compose.yml up -d --build
```

### `pip install -r requirements.txt` failed on EC2

Problem:

```text
externally-managed-environment
```

Fix:

Use a virtual environment:

```bash
sudo apt install -y python3-venv
python3 -m venv .venv
source .venv/bin/activate
```

### Pandas build failed on EC2

Problem:

```text
pandas-2.2.3.tar.gz
cc: fatal error: Killed signal terminated program cc1
```

Cause:

EC2 used Python 3.14, so pip tried to compile pandas from source. The micro instance did not have enough memory.

Fix:

Do not install full app requirements on the host. Install only DVC on the host and let Docker install app dependencies inside `python:3.12-slim`.

## 13. Recreate Deployment From Zero

Short checklist:

1. Create EC2 Ubuntu instance.
2. Open inbound ports 22, 8000, 8501.
3. SSH into EC2.
4. Install git, Docker, Python venv, curl, unzip.
5. Install standalone docker-compose.
6. Install AWS CLI.
7. Configure AWS credentials for `dvc-s3-user`.
8. Clone GitHub repo.
9. Create `.venv`.
10. Install only DVC dependencies:

```bash
pip install "dvc[s3]==3.64.2" "pathspec==0.12.1" dvc-s3
```

11. Pull DVC model artifacts:

```bash
dvc pull models/model.pkl
dvc pull models/pruned_feature_cols.json
```

12. Build and run:

```bash
docker-compose -f docker/docker-compose.yml up -d --build
```

13. Test:

```text
http://EC2_PUBLIC_IP:8000/docs
http://EC2_PUBLIC_IP:8501
```

14. Add/update GitHub secrets for CD:

```text
EC2_HOST
EC2_USERNAME
EC2_SSH_KEY
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
```

15. Push to `main` and confirm both CI and Deploy workflows pass.

## 14. Cost And Cleanup Notes

To avoid unnecessary AWS cost:

- stop EC2 when not demoing
- delete unused EBS volumes after terminating instances
- keep S3 bucket private
- avoid enabling S3 versioning for this learning project unless needed
- avoid repeated large DVC pushes unless artifacts actually changed
- set AWS budget alerts

Cleanup commands on EC2:

```bash
docker ps
docker stop CONTAINER_ID
docker system prune -a
```

Only run prune commands when you are comfortable rebuilding images.

